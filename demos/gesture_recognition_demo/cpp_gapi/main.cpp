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

G_API_NET(PersoneDetection, <cv::GMat(cv::GMat)>, "perspne_detection");

G_API_NET(ActionRecognition, <cv::GMat(cv::GMat)>, "action_recognition");

G_TYPED_KERNEL(PersonTrack, <cv::GOpaque<std::map<size_t, Detection>>(cv::GOpaque<BoundingBoxesSet>)>, "custom.track") {
    static cv::GOpaqueDesc outMeta(const cv::GOpaqueDesc&) {
        return cv::empty_gopaque_desc();
    }
};

// Track persons; See original demo
GAPI_OCV_KERNEL_ST(OCVPersonTrack, PersonTrack, RegisteredPersons) {
    static void setup(const cv::GOpaqueDesc&,
                      std::shared_ptr<RegisteredPersons> &tracked) {
        RegisteredPersons persons = {};
        tracked = std::make_shared<RegisteredPersons>(persons);
    }

    static void run(const BoundingBoxesSet& new_rois,
                    std::map<size_t, Detection>& out_persons,
                    RegisteredPersons &tracked) {
        BoundingBoxesSet filtered_rois = new_rois;

        // If it is the first run of this kernel just fill the state with detections
        if (tracked.active_persons.empty()){
            for(auto it = filtered_rois.begin(); it != filtered_rois.end(); ++it) {
                tracked.active_persons[tracked.last_id] = Detection(*it, 0);
                tracked.last_id++;
            }
        }
        // If we have any detection
        else if(!filtered_rois.empty()) {
            // Find most shapable roi from active persons set
            for(auto it = tracked.active_persons.begin(); it != tracked.active_persons.end(); ){
                float max_shape = 0.;
                BoundingBoxesSet::iterator actual_roi_candidate;
                for(auto roi_it = filtered_rois.begin(); roi_it != filtered_rois.end(); roi_it++){
                    cv::Rect tracked_roi = it->second.roi;
                    cv::Rect candidate_roi = *roi_it;

                    float inter_area = (tracked_roi & candidate_roi).area();
                    float common_area = tracked_roi.area() + candidate_roi.area() - inter_area;
                    float shape = inter_area / common_area;

                    if (shape > IOU_THRESHOLD && shape > max_shape) {
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
            // There is some new persons on frame
            for(auto roi = filtered_rois.begin(); roi != filtered_rois.end(); ){
                // try to find it in the "waiting spot"
                // if it didnt have a match, create new person

                // Find most shapable roi from waiting persons set
                float max_shape = 0.;
                std::map<size_t, Detection>::iterator detection_candidate;
                BoundingBoxesSet::iterator roi_candidate;
                for (auto it = tracked.waiting_persons.begin(); it != tracked.waiting_persons.end(); it++) {
                    cv::Rect waiting_roi = it->second.roi;
                    cv::Rect tracked_roi = *roi;

                    float inter_area = (waiting_roi & tracked_roi).area();
                    float common_area = waiting_roi.area() + tracked_roi.area() - inter_area;
                    float shape = inter_area / common_area;
                    
                    if (shape > IOU_THRESHOLD & shape > max_shape) {
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

        out_persons = tracked.active_persons;
    }
};

G_API_OP(BoundingBoxExtract, <cv::GOpaque<BoundingBoxesSet>(cv::GMat, cv::GMat)>, "custom.bb_extract") {
    static cv::GOpaqueDesc outMeta(const cv::GMatDesc &in, const cv::GMatDesc &) {
        return cv::empty_gopaque_desc();
    }
};

// Just extract persons bounding boxes
GAPI_OCV_KERNEL(OCVBoundingBoxExtract, BoundingBoxExtract) {
    static void run(const cv::Mat &in_ssd_result,
                    const cv::Mat &in_frame,
                    BoundingBoxesSet &bboxes) {
        float scaling_x = in_frame.size().width / PERSON_DETECTOR_W;
        float scaling_y = in_frame.size().height / PERSON_DETECTOR_H;

        bboxes.clear();
        const float *data = in_ssd_result.ptr<float>();
        for (int i =0; i < 100; i++) {
            const int OBJECT_SIZE = 5;

            const float x_min = data[i * OBJECT_SIZE + 0];
            const float y_min = data[i * OBJECT_SIZE + 1];
            const float x_max = data[i * OBJECT_SIZE + 2];
            const float y_max = data[i * OBJECT_SIZE + 3];
            const float conf  = data[i * OBJECT_SIZE + 4];

            if (conf > BOUNDING_BOX_THRESHOLD)
                bboxes.insert(cv::Rect(static_cast<int>(x_min * scaling_x),
                                          static_cast<int>(y_min * scaling_y),
                                          static_cast<int>((x_max - x_min) * scaling_x),
                                          static_cast<int>((y_max - y_min) * scaling_y)));

        }
    }
};

G_TYPED_KERNEL(ActionRecognisePreprocessing, <cv::GArray<cv::GMat>(const cv::GMat, const cv::GOpaque<std::map<size_t, Detection>>/*SOme more inputs??*/)>,
                        "custom.frames_storage") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc&, const cv::GOpaqueDesc&) {
        return cv::empty_array_desc();
    }
};

// Prepare input for gesture recognition person
// cut all roi`s from batch, resize, split and aquash it in to a one
// [1, 3, 16, 224, 224] cv::Mat
GAPI_OCV_KERNEL_ST(OCVPreprocessing, ActionRecognisePreprocessing, std::list<cv::Mat>) {
    static void setup(const cv::GMatDesc&, const cv::GOpaqueDesc&,
                      std::shared_ptr<std::list<cv::Mat>> &stored_frames) {
        std::list<cv::Mat> frames = {};
        stored_frames = std::make_shared<std::list<cv::Mat>>(frames);
    }

    static void run(const cv::Mat &in_frame,
                    const std::map<size_t, Detection> &tracked_persons,
                    std::vector<cv::Mat> &prepared_mat, 
                    std::list<cv::Mat> &stored_frames) {
        if (stored_frames.size() < 16) {
            stored_frames.push_back(in_frame);
        }
        else {
            if(prepared_mat.size() == 0) {
                cv::Mat mat_to_process;
                mat_to_process.create({1, 3, 16, 224, 224}, CV_32F);
                prepared_mat.push_back(mat_to_process);
            }

            stored_frames.pop_front();
            stored_frames.push_back(in_frame);

            int i = 0;
            for(auto frame = stored_frames.begin(); frame != stored_frames.end(); ++frame) {
                cv::Rect roi = tracked_persons.begin()->second.roi;
                cv::Mat crop = (*frame)(roi);
                crop.convertTo(crop, CV_32F);

                cv::Mat resized;
                int H = 224, W = 224; // Take sizes from gesture recognition net description
                cv::Size sz_dst(H, W);
                cv::resize(crop, resized, sz_dst);

                cv::Mat different_channels[3];
                cv::split(resized, different_channels);
                cv::Mat matB = different_channels[0];
                cv::Mat matG = different_channels[1];
                cv::Mat matR = different_channels[2];

                std::copy_n(matR.begin<float>(), H * W, prepared_mat[0].begin<float>() + 0 * 16 * H * W
                                                                                        + i * H * W);
                std::copy_n(matG.begin<float>(), H * W, prepared_mat[0].begin<float>() + 1 * 16 * H * W
                                                                                        + i * H * W);
                std::copy_n(matB.begin<float>(), H * W, prepared_mat[0].begin<float>() + 2 * 16 * H * W
                                                                                        + i * H * W);
                ++i;
            }
        }
    }
};

int main(int argc, char *argv[]) {
    try {
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        cv::GComputation pipeline([=]() {
                cv::GMat in;
                cv::GMat out_frame = cv::gapi::copy(in);

                cv::GMat detections = cv::gapi::infer<PersoneDetection>(in);

                cv::GOpaque<BoundingBoxesSet> filtered = BoundingBoxExtract::on(detections, in);

                cv::GOpaque<std::map<size_t, Detection>> tracked = PersonTrack::on(filtered);

                cv::GArray<cv::GMat> batch = ActionRecognisePreprocessing::on(in, tracked);

                cv::GArray<cv::GMat> result = cv::gapi::infer2<ActionRecognition>(in, batch);

                return cv::GComputation(cv::GIn(in), cv::GOut( out_frame, tracked, result));
        });

        auto person_detection = cv::gapi::ie::Params<PersoneDetection> {
            FLAGS_m_d,                         // path to model
            fileNameNoExt(FLAGS_m_d) + ".bin", // path to weights
            "CPU"                              // device to use
        }.cfgOutputLayers({"boxes"}); // This clarification here because
                                      // of GAPI take the last layer from .xml
                                      // and last layer sould be the outpul layer

        auto action_recognition = cv::gapi::ie::Params<ActionRecognition> {
            FLAGS_m_a,                         // path to model
            fileNameNoExt(FLAGS_m_a) + ".bin", // path to weights
            "CPU"                              // device to use
        }.cfgOutputLayers({"output"}); // Same 

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
        int last_action = 0;
        // Main cycle
        while (stream.pull(std::move(out_vector))){
                // Draw any tracked person
                for(auto person_id_roi : detections){
                    int id = person_id_roi.first;
                    cv::Rect bb = person_id_roi.second.roi;

                    cv::putText(frame, std::to_string(id),
                                cv::Point(bb.x, bb.y), 
                                cv::FONT_HERSHEY_SIMPLEX,
                                1, cv::Scalar(0, 255, 0));
                    cv::rectangle(frame, bb, cv::Scalar(0, 255, 0));
                }

                // Draw main person
                // In this realisation main person is just a 
                // person with a least id
                int id = detections.begin()->first;
                cv::Rect bb = detections.begin()->second.roi;

                cv::putText(frame, std::to_string(id),
                            cv::Point(bb.x, bb.y), 
                            cv::FONT_HERSHEY_SIMPLEX,
                            1, cv::Scalar(0, 0, 255));
                cv::rectangle(frame, bb, cv::Scalar(0, 0, 255));


                // Draw action
                if (!actions_weights.empty()) {
                    const float* data = actions_weights[0].ptr<float>();

                    // Find more suitabl action
                    float max_weight = 0.;
                    for(int i = 0; i < NUM_CLASSES; i++) {
                        if (data[i] > max_weight && data[i] > AR_THRASHOLD) {
                            max_weight = data[i];
                            last_action = i;
                        }
                    }

                    cv::putText(frame, ACTIONS_MAP[last_action],
                                cv::Point(0, frame.size[1]), 
                                cv::FONT_HERSHEY_SIMPLEX,
                                1, cv::Scalar(0, 0, 255));
                }

                if (!videoWriter.isOpened()) {
                    videoWriter.open(FLAGS_o, cv::VideoWriter::fourcc('I', 'Y', 'U', 'V'), 25, cv::Size(frame.size()));
                }

                if (!FLAGS_o.empty()) {
                    videoWriter.write(frame);
                }
        }
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
