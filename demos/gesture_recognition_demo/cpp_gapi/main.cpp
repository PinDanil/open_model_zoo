// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the gesture recognition
* \file gesture_recognition_gapi/main.cpp
* \example gesture_recognition_gapi/main.cpp
*/

#include <iostream>
#include <unordered_set>
#include "gesture_recognition_gapi.hpp"
#include <utils/slog.hpp>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

#include <opencv2/highgui.hpp>

#include "utils.hpp"

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        // showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;
    if (FLAGS_i.empty())
        throw std::logic_error("Parameter -i is not set");
    if (FLAGS_o.empty())
        throw std::logic_error("Parameter -o is not set");
    if (FLAGS_m_a.empty())
        throw std::logic_error("Parameter -am is not set");
    if (FLAGS_m_d.empty())
        throw std::logic_error("Parameter -dm is not set");
    if (FLAGS_c.empty())
        throw std::logic_error("Parameter -c is not set");

    return true;
}

void setInput(cv::GStreamingCompiled stream, const std::string& input ) {
    try {
        // If stoi() throws exception input should be a path not a camera id
        stream.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(std::stoi(input)));
    } catch (std::invalid_argument&) {
        slog::info << "Input source is treated as a file path" << slog::endl;
        stream.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input));
    }
}

G_API_NET(PersoneDetection, <cv::GMat(cv::GMat)>, "perspne_detection");

const float TRACKER_SCORE_THRESHOLD = 0.4;
const float TRACKER_IOU_THRESHOLD = 0.3;
const int   WAITING_PERSON_DURATION = 8;

struct Detection
{
    cv::Rect roi;
    int waiting = 1;

    Detection(const cv::Rect& r = cv::Rect(), const int w = 0):
        roi(r), waiting(w){}
};

struct StateMap {
    std::map<size_t, Detection> active_persons;
    std::map<size_t, Detection> waiting_persons;
    int last_id = 0;
};

using RectSet = std::unordered_set<cv::Rect, rectHash, rectEqual>;

G_TYPED_KERNEL(PersonTrack, <cv::GOpaque<std::map<size_t, Detection>>(cv::GOpaque<RectSet>)>, "custom.track") {
    static cv::GOpaqueDesc outMeta(const cv::GOpaqueDesc&) {
        return cv::empty_gopaque_desc();
    }
};

GAPI_OCV_KERNEL_ST(OCVPersonTrack, PersonTrack, StateMap) {
    static void setup(const cv::GOpaqueDesc&,
                      std::shared_ptr<StateMap> &tracked) {
        StateMap persons = {};
        tracked = std::make_shared<StateMap>(persons);
    }

    static void run(const RectSet& new_rois,
                    std::map<size_t, Detection>& out_persons,
                    StateMap &tracked) {
        RectSet filtered_rois = new_rois;

        if (tracked.active_persons.empty()){
            for(auto it = filtered_rois.begin(); it != filtered_rois.end(); ++it) {
                tracked.active_persons[tracked.last_id] = Detection(*it, 0);
                tracked.last_id++;
            }
        }
        else if(!filtered_rois.empty()) {
            // Find most shapable roi
            for(auto it = tracked.active_persons.begin(); it != tracked.active_persons.end(); ){
                float max_shape = 0.;
                RectSet::iterator actual_roi_candidate;
                for(auto roi_it = filtered_rois.begin(); roi_it != filtered_rois.end(); roi_it++){
                    cv::Rect tracked_roi = it->second.roi;
                    cv::Rect candidate_roi = *roi_it;

                    float inter_area = (tracked_roi & candidate_roi).area();
                    float common_area = tracked_roi.area() + candidate_roi.area() - inter_area;
                    float shape = inter_area / common_area;
                    
                    if (shape > TRACKER_IOU_THRESHOLD & shape > max_shape) {
                        max_shape = shape;
                        actual_roi_candidate = roi_it;
                    }
                }
                if(max_shape != 0.) {
                    it->second.roi = *actual_roi_candidate;
                    filtered_rois.erase(actual_roi_candidate);

                    ++it;
                }
                else { // Didn`t find any shapable roi
                    // move that roi/detectoin to "waiting spot"
                    tracked.waiting_persons[it->first] = it->second;
                    tracked.waiting_persons[it->first].waiting = 1; 
                    tracked.active_persons.erase(it++);
                }
            }
            if(!filtered_rois.empty()){ // There is some new persons on frame
                for(auto roi : filtered_rois){
                    // try to find it in the "waiting spot"
                    // if it didnt hame a match, create new roi
                    tracked.active_persons[tracked.last_id] = {roi, 0};
                    tracked.last_id++;
                }
            }
        }

        out_persons = tracked.active_persons;
    }
};

G_API_OP(BoundingBoxExtract, <cv::GOpaque<RectSet>(cv::GMat, cv::GMat)>, "custom.bb_extract") {
    static cv::GOpaqueDesc outMeta(const cv::GMatDesc &in, const cv::GMatDesc &) {
        return cv::empty_gopaque_desc();
    }
};

GAPI_OCV_KERNEL(OCVBoundingBoxExtract, BoundingBoxExtract) {
    static void run(const cv::Mat &in_ssd_result,
                    const cv::Mat &in_frame,
                    RectSet &bboxes) {
        // get scaling params [320, 320] before
        float scaling_x = in_frame.size().width / 320.;
        float scaling_y = in_frame.size().height / 320.;

        bboxes.clear();
        const float *data = in_ssd_result.ptr<float>();
        for (int i =0; i < 100; i++) {
            const int OBJECT_SIZE = 5;

            const float x_min = data[i * OBJECT_SIZE + 0];
            const float y_min = data[i * OBJECT_SIZE + 1];
            const float x_max = data[i * OBJECT_SIZE + 2];
            const float y_max = data[i * OBJECT_SIZE + 3];
            const float conf  = data[i * OBJECT_SIZE + 4];

            if (conf > TRACKER_SCORE_THRESHOLD){
                cv::Rect boundingBox(
                    static_cast<int>(x_min * scaling_x),
                    static_cast<int>(y_min * scaling_y),
                    static_cast<int>((x_max - x_min) * scaling_x),
                    static_cast<int>((y_max - y_min) * scaling_y)
                );

                bboxes.insert(cv::Rect(static_cast<int>(x_min * scaling_x),
                                          static_cast<int>(y_min * scaling_y),
                                          static_cast<int>((x_max - x_min) * scaling_x),
                                          static_cast<int>((y_max - y_min) * scaling_y)));

            }
        }
    }
};

static std::string fileNameNoExt(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath;
    return filepath.substr(0, pos);
}

int main(int argc, char *argv[]) {
    try {
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
        // Load network and set input

        cv::GComputation pipeline([=]() {
                cv::GMat in;
                cv::GMat out_frame = cv::gapi::copy(in);

                cv::GMat detections = cv::gapi::infer<PersoneDetection>(in);

                cv::GOpaque<RectSet> filtered = BoundingBoxExtract::on(detections, in);

                cv::GOpaque<std::map<size_t, Detection>> tracked = PersonTrack::on(filtered);

                return cv::GComputation(cv::GIn(in), cv::GOut(tracked, out_frame));
        });

        auto person_detection = cv::gapi::ie::Params<PersoneDetection> {
            FLAGS_m_d,                         // path to model
            fileNameNoExt(FLAGS_m_d) + ".bin", // path to weights
            "CPU"                              // device to use
        }.cfgOutputLayers({"boxes"});

        auto kernels = cv::gapi::kernels<OCVBoundingBoxExtract, OCVPersonTrack>();
        auto networks = cv::gapi::networks(person_detection);

        cv::VideoWriter videoWriter;
        cv::Mat frame;
        std::map<size_t, Detection> detections;
        auto out_vector = cv::gout(detections, frame);

        std::vector<cv::Rect> bbDetections;

        auto stream = pipeline.compileStreaming(cv::compile_args(kernels, networks));
        setInput(stream, FLAGS_i);
        stream.start();
        while (stream.pull(std::move(out_vector))){
                // std::cout<< "Size of map : "<< detections.size()<<std::endl;
                for(auto person_pair : detections){
                    int id = person_pair.first;
                    cv::Rect bb = person_pair.second.roi;

                    cv::putText(frame, std::to_string(id),
                                cv::Point(bb.x, bb.y), 
                                cv::FONT_HERSHEY_SIMPLEX,
                                1, cv::Scalar(0, 255, 0));
                    cv::rectangle(frame, bb, cv::Scalar(0, 255, 0));
                }
                // std::cout<< 1 <<std::endl;

                if (!videoWriter.isOpened()) {
                    videoWriter.open(FLAGS_o, cv::VideoWriter::fourcc('I', 'Y', 'U', 'V'), 25, cv::Size(frame.size()));
                }
                // std::cout<< 2 <<std::endl;

                if (!FLAGS_o.empty()) {
                    videoWriter.write(frame);
                }
                // std::cout<< 3 <<std::endl;

        }
        std::cout<<"End of proramm"<< std::endl;
        // ------------------------------ Parsing and validating of input arguments --------------------------
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }
    slog::info << "Execution successful" << slog::endl;

    return 0;
}
