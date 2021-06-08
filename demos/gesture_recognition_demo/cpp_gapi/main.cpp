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

#include <list>

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

const float TRACKER_SCORE_THRESHOLD = 0.4;
const float TRACKER_IOU_THRESHOLD = 0.3;
const int   WAITING_PERSON_DURATION = 8;

G_API_NET(PersoneDetection, <cv::GMat(cv::GMat)>, "perspne_detection");

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
                    tracked.waiting_persons[it->first].waiting = 0;
                    tracked.active_persons.erase(it++);
                }
            }
            if(!filtered_rois.empty()) {
                // There is some new persons on frame
                for(auto roi = filtered_rois.begin(); roi != filtered_rois.end(); ){
                    // try to find it in the "waiting spot"
                    // if it didnt hame a match, create new roi
                    
                    // Finding person in waiting_persons
                    float max_shape = 0.;
                    std::map<size_t, Detection>::iterator detection_candidate;
                    RectSet::iterator roi_candidate;
                    for (auto it = tracked.waiting_persons.begin(); it != tracked.waiting_persons.end(); it++) {
                        cv::Rect waiting_roi = it->second.roi;
                        cv::Rect tracked_roi = *roi;

                        float inter_area = (waiting_roi & tracked_roi).area();
                        float common_area = waiting_roi.area() + tracked_roi.area() - inter_area;
                        float shape = inter_area / common_area;
                        
                        if (shape > TRACKER_IOU_THRESHOLD & shape > max_shape) {
                            max_shape = shape;
                            detection_candidate = it;
                            roi_candidate = roi;
                        }
                    }

                    if(max_shape != 0.) {
                        // Move roi to active person
                        int id = detection_candidate->first;
                        auto value = *roi;
                        tracked.active_persons[id] = Detection(value);
                        
                        tracked.waiting_persons.erase(detection_candidate);
                        filtered_rois.erase(roi++);
                    } else {
                        ++roi;
                    }
                }

                for (auto roi : filtered_rois){
                    tracked.active_persons[tracked.last_id] = {roi, 0};
                    tracked.last_id++;
                }
            }
        }

        for(auto waiting_person =  tracked.waiting_persons.begin();
                 waiting_person != tracked.waiting_persons.end(); ){
            waiting_person->second.waiting++;

            if(waiting_person->second.waiting > WAITING_PERSON_DURATION){
                tracked.waiting_persons.erase(waiting_person++);
            }
            else {
                ++waiting_person;
            }
        }
        // Increse waitin values;

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

G_TYPED_KERNEL(Preprocessing, <cv::GArray<cv::GMat>(const cv::GMat, const cv::GOpaque<std::map<size_t, Detection>>/*SOme more inputs??*/)>,
                        "custom.frames_storage") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc&, const cv::GOpaqueDesc&) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL_ST(OCVPreprocessing, Preprocessing, std::list<cv::Mat>) {
    static void setup(const cv::GMatDesc&, const cv::GOpaqueDesc&,
                      std::shared_ptr<std::list<cv::Mat>> &stored_frames) {
        std::list<cv::Mat> frames = {};
        stored_frames = std::make_shared<std::list<cv::Mat>>(frames);
    }

    static void run(const cv::Mat &in_frame,
                    const std::map<size_t, Detection> &tracked_persons,
                    std::vector<cv::Mat> &prepared_mat, 
                    std::list<cv::Mat> &stored_frames) {
        stored_frames.clear();
        if ( stored_frames.size() < 16) {
            stored_frames.push_back(in_frame);
        }
        else {
            stored_frames.pop_front();
            stored_frames.push_back(in_frame);

            int i = 0;
            for(auto frame = stored_frames.begin(); frame != stored_frames.end(); ++frame) {
                // Take roi from tracked_persons
                // cv::Rect roi = tracked_persons.begin()->second.roi;
                cv::Rect roi = tracked_persons.begin()->second.roi;
                cv::Mat crop = (*frame)(roi);

                cv::Mat resized;
                int H = 224, W = 224; // Take sizes from net description
                cv::Size sz_dst(H, W);
                cv::resize(crop, resized, sz_dst);

                // do stuff from there
                cv::Mat different_channels[3];

                cv::split(resized, different_channels);
                cv::Mat matB = different_channels[0];
                cv::Mat matG = different_channels[1];
                cv::Mat matR = different_channels[2];

                std::copy_n(matB.begin<uint8_t>(), H * W, prepared_mat[0].begin<uint8_t>() + 0 * 16 * H * W
                                                                                        + i * H * W);
                std::copy_n(matG.begin<uint8_t>(), H * W, prepared_mat[0].begin<uint8_t>() + 1 * 16 * H * W
                                                                                        + i * H * W);
                std::copy_n(matR.begin<uint8_t>(), H * W, prepared_mat[0].begin<uint8_t>() + 2 * 16 * H * W
                                                                                        + i * H * W);
                ++i;
            }
        }
    }
};

G_API_NET(ActionRecognition, <cv::GMat(cv::GMat)>, "action_recognition");

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

                cv::GArray<cv::GMat> batch = Preprocessing::on(in, tracked);

                cv::GArray<cv::GMat> result = cv::gapi::infer2<ActionRecognition>(in, batch);

                return cv::GComputation(cv::GIn(in), cv::GOut( out_frame, tracked, result));
        });

        auto person_detection = cv::gapi::ie::Params<PersoneDetection> {
            FLAGS_m_d,                         // path to model
            fileNameNoExt(FLAGS_m_d) + ".bin", // path to weights
            "CPU"                              // device to use
        }.cfgOutputLayers({"boxes"});

        auto action_recognition = cv::gapi::ie::Params<ActionRecognition> {
            FLAGS_m_a,                         // path to model
            fileNameNoExt(FLAGS_m_a) + ".bin", // path to weights
            "CPU"                              // device to use
        };

        auto kernels = cv::gapi::kernels<OCVBoundingBoxExtract, OCVPersonTrack, OCVPreprocessing>();
        auto networks = cv::gapi::networks(person_detection, action_recognition);

        cv::VideoWriter videoWriter;
        cv::Mat frame;
        std::map<size_t, Detection> detections;
        std::vector<cv::Mat> actions_weights;
        size_t id;
        auto out_vector = cv::gout(frame, detections, actions_weights);

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

                {
                    int id = detections.begin()->first;
                    cv::Rect bb = detections.begin()->second.roi;

                    cv::putText(frame, std::to_string(id),
                                cv::Point(bb.x, bb.y), 
                                cv::FONT_HERSHEY_SIMPLEX,
                                1, cv::Scalar(0, 0, 255));
                    cv::rectangle(frame, bb, cv::Scalar(0, 0, 255));

                    int max_action_value = 0;
                    float max_weight = 0.;

                    std::cout<< "Sizes: " << actions_weights.size()<<std::endl;
/*
                    for(size_t i = 0 ; i < actions_weights.size(); i++) {
                        if (actions_weights[i] > max_weight) {
                            max_weight = actions_weights[i];
                            max_action_value = i;
                        }
                    }
*/
                    cv::putText(frame, ACTIONS_MAP[max_action_value],
                                cv::Point(0, 0), 
                                cv::FONT_HERSHEY_SIMPLEX,
                                1, cv::Scalar(0, 0, 255));
                }

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
