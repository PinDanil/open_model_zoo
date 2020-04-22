// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine interactive_face_detection demo application
* \file interactive_face_detection_demo/main.cpp
* \example interactive_face_detection_demo/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>
#include <list>
#include <set>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include "gapi_stuff.hpp"


#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/streaming/cap.hpp>


#include "interactive_face_detection.hpp"
#include "detectors.hpp"
#include "face.hpp"
#include "visualizer.hpp"

#include <ie_iextension.h>
#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif

using namespace InferenceEngine;


bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_n_ag < 1) {
        throw std::logic_error("Parameter -n_ag cannot be 0");
    }

    if (FLAGS_n_hp < 1) {
        throw std::logic_error("Parameter -n_hp cannot be 0");
    }

    // no need to wait for a key press from a user if an output image/video file is not shown.
    FLAGS_no_wait |= FLAGS_no_show;

    return true;
}

int main(int argc, char *argv[]) {
    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validating of input arguments --------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
        slog::info << "Start inference " << slog::endl;

        std::ostringstream out;
        size_t framesCounter = 0;
        int delay = 1;
        double msrate = -1;
        cv::Mat prev_frame, next_frame;
        std::list<Face::Ptr> faces;
        size_t id = 0;

        if (FLAGS_fps > 0) {
            msrate = 1000.f / FLAGS_fps;
        }

        std::cout << "To close the application, press 'CTRL+C' here";
        if (!FLAGS_no_show) {
            std::cout << " or switch to the output window and press any key";
        }
        std::cout << std::endl;

        G_API_NET(Faces, <cv::GMat(cv::GMat)>, "face-detector");

        using AGInfo = std::tuple<cv::GMat, cv::GMat>;
        G_API_NET(AgeGender, <AGInfo(cv::GMat)>,   "age-gender-recoginition");

        using HPInfo = std::tuple<cv::GMat, cv::GMat, cv::GMat>;
        G_API_NET(HeadPose, <HPInfo(cv::GMat)>,   "head-pose-recoginition");

        G_API_NET(FacialLandmark, <cv::GMat(cv::GMat)>,   "facial-landmark-recoginition");

        G_API_NET(Emotions, <cv::GMat(cv::GMat)>, "emotions-recognition");

        G_API_OP(PostProc, <cv::GArray<cv::Rect>(cv::GMat, cv::GMat)>, "custom.fd_postproc") {
            static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GMatDesc &) {
                return cv::empty_array_desc();
            }
        };

        GAPI_OCV_KERNEL(OCVPostProc, PostProc) {
            static void run(const cv::Mat &in_ssd_result,
                            const cv::Mat &in_frame,
                            std::vector<cv::Rect> &out_faces) {
                const int MAX_PROPOSALS = 200;
                const int OBJECT_SIZE   =   7;
                const cv::Size upscale = in_frame.size();
                const cv::Rect surface({0,0}, upscale);

                out_faces.clear();

                const float *data = in_ssd_result.ptr<float>();
                for (int i = 0; i < MAX_PROPOSALS; i++) {
                    const float image_id   = data[i * OBJECT_SIZE + 0]; // batch id
                    const float confidence = data[i * OBJECT_SIZE + 2];
                    const float rc_left    = data[i * OBJECT_SIZE + 3];
                    const float rc_top     = data[i * OBJECT_SIZE + 4];
                    const float rc_right   = data[i * OBJECT_SIZE + 5];
                    const float rc_bottom  = data[i * OBJECT_SIZE + 6];

                    if (image_id < 0.f) {  // indicates end of detections
                        break;
                    }
                    if (confidence < 0.5f) { // fixme: hard-coded snapshot
                        continue;
                    }

                    cv::Rect rc;
                    rc.x      = static_cast<int>(rc_left   * upscale.width);
                    rc.y      = static_cast<int>(rc_top    * upscale.height);
                    rc.width  = static_cast<int>(rc_right  * upscale.width)  - rc.x;
                    rc.height = static_cast<int>(rc_bottom * upscale.height) - rc.y;

                    // Make square and enlarge face bounding box for more robust operation of face analytics networks
                    int bb_width = rc.width;
                    int bb_height = rc.height;

                    int bb_center_x = rc.x + bb_width / 2;
                    int bb_center_y = rc.y + bb_height / 2;

                    int max_of_sizes = std::max(bb_width, bb_height);
            
                    //bb_enlarge_coefficient, dx_coef, dy_coef is a omz flags
                    //usualy it's 1.2, 1.0 and 1.0
                    float bb_enlarge_coefficient = 1.2;
                    float bb_dx_coefficient = 1.0;
                    float bb_dy_coefficient = 1.0;
                    int bb_new_width = static_cast<int>(bb_enlarge_coefficient * max_of_sizes);
                    int bb_new_height = static_cast<int>(bb_enlarge_coefficient * max_of_sizes);

                    rc.x = bb_center_x - static_cast<int>(std::floor(bb_dx_coefficient * bb_new_width / 2));
                    rc.y = bb_center_y - static_cast<int>(std::floor(bb_dy_coefficient * bb_new_height / 2));

                    rc.width = bb_new_width;
                    rc.height = bb_new_height;

                    out_faces.push_back(rc & surface);
                }
            }
        };

        cv::GComputation pipeline([]() {
                cv::GMat in;

                cv::GMat detections = cv::gapi::infer<Faces>(in);

                cv::GArray<cv::Rect> faces = PostProc::on(detections, in);

                cv::GArray<cv::GMat> ages;
                cv::GArray<cv::GMat> genders;
                std::tie(ages, genders) = cv::gapi::infer<AgeGender>(faces, in);

                cv::GArray<cv::GMat> y_fc;
                cv::GArray<cv::GMat> p_fc;
                cv::GArray<cv::GMat> r_fc;
                std::tie(y_fc, p_fc, r_fc) = cv::gapi::infer<HeadPose>(faces, in);

                cv::GArray<cv::GMat> landmarks = cv::gapi::infer<FacialLandmark>(faces, in);

                cv::GArray<cv::GMat> emotions = cv::gapi::infer<Emotions>(faces, in);

                cv::GMat frame = cv::gapi::copy(in);

                cv::GProtoOutputArgs outs = GOut(frame);
                outs += GOut(faces);
                outs += GOut(detections);
                outs += GOut(ages, genders); 
                outs += GOut(y_fc, p_fc, r_fc);
                outs += GOut(emotions);
                outs += GOut(landmarks);

                return cv::GComputation(cv::GIn(in),
                                        cv::GOut(frame,
                                                 faces,
                                                 detections, ages, genders, 
                                                 y_fc, p_fc, r_fc,
                                                 emotions,
                                                 landmarks));
        });

        auto det_net = cv::gapi::ie::Params<Faces> {
            FLAGS_m,   // read cmd args: path to topology IR
            FLAGS_w,   // read cmd args: path to weights
            FLAGS_d,   // read cmd args: device specifier
        };

        auto age_net = cv::gapi::ie::Params<AgeGender> {
            FLAGS_m_ag,   // read cmd args: path to topology IR
            FLAGS_w_ag,   // read cmd args: path to weights
            FLAGS_d_ag,   // read cmd args: device specifier
        }.cfgOutputLayers({ "age_conv3", "prob" });

        auto hp_net = cv::gapi::ie::Params<HeadPose> {
            FLAGS_m_hp,   // read cmd args: path to topology IR
            FLAGS_w_hp,   // read cmd args: path to weights
            FLAGS_d_hp,   // read cmd args: device specifier
        }.cfgOutputLayers({ "angle_y_fc", "angle_p_fc", "angle_r_fc" });

        auto lm_net = cv::gapi::ie::Params<FacialLandmark> {
            FLAGS_m_lm,   // read cmd args: path to topology IR
            FLAGS_w_lm,   // read cmd args: path to weights
            FLAGS_d_lm,   // read cmd args: device specifier
        };

        auto emo_net = cv::gapi::ie::Params<Emotions> {
            FLAGS_m_em,   // read cmd args: path to topology IR
            FLAGS_w_em,   // read cmd args: path to weights
            FLAGS_d_em,   // read cmd args: device specifier
        };

        // Form a kernel package (with a single OpenCV-based implementation of our
        // post-processing) and a network package (holding our three networks).x
        auto kernels = cv::gapi::kernels<OCVPostProc>();
        auto networks = cv::gapi::networks(det_net, age_net, hp_net, lm_net, emo_net);

        cv::GStreamingCompiled stream = pipeline.compileStreaming(cv::compile_args(kernels, networks));

        cv::Mat frame;
        cv::Mat out_detections;
        std::vector<cv::Rect> face_hub;
        std::vector<cv::Mat> out_ages;
        std::vector<cv::Mat> out_genders;
        std::vector<cv::Mat> out_y_fc, out_p_fc, out_r_fc; 
        std::vector<cv::Mat> out_landmarks;
        std::vector<cv::Mat> out_emotions;
    
        stream.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(FLAGS_i));

        cv::GRunArgsP out_vector = cv::gout(frame, face_hub, 
                                            out_detections,
                                            out_ages, out_genders,
                                            out_y_fc, out_p_fc, out_r_fc,
                                            out_emotions, out_landmarks);


        cv::namedWindow("Detection results");

        stream.start();
        stream.pull(std::move(out_vector));

        const size_t width  = static_cast<size_t>(frame.cols);
        const size_t height = static_cast<size_t>(frame.rows);

        Timer timer;
 
        Visualizer::Ptr visualizer;
        if (!FLAGS_no_show || !FLAGS_o.empty()) {
            visualizer = std::make_shared<Visualizer>(cv::Size(width, height));
            if (!FLAGS_no_show_emotion_bar /*&& emotionsDetector.enabled()*/) {
                visualizer->enableEmotionBar({"neutral",
                                              "happy",
                                              "sad",
                                              "surprise",
                                              "anger"});
            }
        }

        while (stream.running())
        {
            timer.start("total");
        
            stream.pull(std::move(out_vector));

            if (!FLAGS_no_show && -1 != cv::waitKey(delay)) {
                stream.stop();
            }

            //  Postprocessing
            std::list<Face::Ptr> prev_faces;

            if (!FLAGS_no_smooth) {
                prev_faces.insert(prev_faces.begin(), faces.begin(), faces.end());
            }

            faces.clear();

            // For every detected face
            for (size_t i = 0; i < face_hub.size(); i++) {
                //auto& result = prev_detection_results[i];
                cv::Rect rect = face_hub[i] & cv::Rect({0, 0}, frame.size());

                Face::Ptr face;
                // wthat is this for??
                if (!FLAGS_no_smooth) {
                    face = matchFace(rect, prev_faces);
                    float intensity_mean = calcMean(frame(rect));
                    intensity_mean += 1.0;
            
                    if ((face == nullptr) ||
                        ((face != nullptr) && ((std::abs(intensity_mean - face->_intensity_mean) / face->_intensity_mean) > 0.07f))) {
                        face = std::make_shared<Face>(id++, rect);
                    } else {
                        prev_faces.remove(face);
                    }
            
                    face->_intensity_mean = intensity_mean;
                    face->_location = rect;
                } else {
                    face = std::make_shared<Face>(id++, rect);
                }

                face->ageGenderEnable(/*(ageGenderDetector.enabled() &&
                                       i < ageGenderDetector.maxBatch)*/
                                       true);            
                face->updateGender(out_genders[i].at<float>(0));
                face->updateAge(out_ages[i].at<float>(0) * 100);


                face->headPoseEnable(/*(headPoseDetector.enabled() &&
                                      i < headPoseDetector.maxBatch)*/true);
                if (/*face->isHeadPoseEnabled()*/ true) {
                    face->updateHeadPose({out_r_fc[i].at<float>(0),
                                          out_p_fc[i].at<float>(0),
                                          out_y_fc[i].at<float>(0)});
                }

                face->emotionsEnable(/*(emotionsDetector.enabled() &&
                                  i < emotionsDetector.maxBatch)*/ true);
                face->updateEmotions({
                                      {"neutral", out_emotions[i].at<float>(0)},
                                      {"happy", out_emotions[i].at<float>(1)} ,
                                      {"sad", out_emotions[i].at<float>(2)} ,
                                      {"surprise", out_emotions[i].at<float>(3)}, 
                                      {"anger", out_emotions[i].at<float>(4)}
                                      });

                face->landmarksEnable(/*(facialLandmarksDetector.enabled() &&
                                       i < facialLandmarksDetector.maxBatch)*/ true);
                std::vector<float> normedLandmarks;
                int n_lm = 70;
                for (auto i_lm = 0; i_lm < n_lm; ++i_lm) {
                    float normed_x = out_landmarks[i].at<float>(2 * i_lm);
                    float normed_y = out_landmarks[i].at<float>(2 * i_lm + 1);

                    normedLandmarks.push_back(normed_x);
                    normedLandmarks.push_back(normed_y);
                }

                face->updateLandmarks(normedLandmarks);

                // End of face postprocesing

                faces.push_back(face);            
            }

            //  Visualizing results
            if (!FLAGS_no_show || !FLAGS_o.empty()) {
                out.str("");
                out << "Total image throughput: " << std::fixed << std::setprecision(2)
                    << 1000.f / (timer["total"].getSmoothedDuration()) << " fps";
                cv::putText(frame, out.str(), cv::Point2f(10, 45), cv::FONT_HERSHEY_TRIPLEX, 1.2,
                            cv::Scalar(255, 0, 0), 2);

                // drawing faces
                visualizer->draw(frame, faces);

                if (!FLAGS_no_show) {
                    cv::imshow("Detection results", frame);
                }
            }

            framesCounter++;

            timer.finish("total");

            if (FLAGS_fps > 0) {
                delay = std::max(1, static_cast<int>(msrate - timer["total"].getLastCallDuration()));
            }
        }

        cv::destroyAllWindows();
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
