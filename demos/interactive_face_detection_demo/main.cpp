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

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/streaming/cap.hpp>


#include "interactive_face_detection.hpp"
#include "utils.hpp"
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

    return true;
}

G_API_NET(Faces, <cv::GMat(cv::GMat)>, "face-detector");

using AGInfo = std::tuple<cv::GMat, cv::GMat>;
G_API_NET(AgeGender, <AGInfo(cv::GMat)>,   "age-gender-recoginition");

using HPInfo = std::tuple<cv::GMat, cv::GMat, cv::GMat>;
G_API_NET(HeadPose, <HPInfo(cv::GMat)>,   "head-pose-recoginition");

G_API_NET(FacialLandmark, <cv::GMat(cv::GMat)>,   "facial-landmark-recoginition");

G_API_NET(Emotions, <cv::GMat(cv::GMat)>, "emotions-recognition");

G_API_OP(PostProc, <cv::GArray<cv::Rect>(cv::GMat, cv::GMat, float)>, "custom.fd_postproc") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GMatDesc &, int) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVPostProc, PostProc) {
    static void run(const cv::Mat &in_ssd_result,
                    const cv::Mat &in_frame,
                    float th,
                    std::vector<cv::Rect> &out_faces) {
        const auto &in_ssd_dims = in_ssd_result.size;
        CV_Assert(in_ssd_dims.dims() == 4u);

        const int MAX_PROPOSALS = in_ssd_dims[2];
        const int OBJECT_SIZE   = in_ssd_dims[3];
        CV_Assert(OBJECT_SIZE == 7);

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
            if (confidence < th) { // fixme: hard-coded snapshot
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
        cv::Mat prev_frame, next_frame;
        std::list<Face::Ptr> faces;
        size_t id = 0;

        std::cout << "To close the application, press 'CTRL+C' here";
        if (!FLAGS_no_show) {
            std::cout << " or switch to the output window and press any key";
        }
        std::cout << std::endl;

        bool age_gender_enable = !FLAGS_m_ag.empty();
        bool headpose_enable   = !FLAGS_m_hp.empty();
        bool emotions_enable   = !FLAGS_m_em.empty();
        bool landmarks_enable  = !FLAGS_m_lm.empty();

        cv::GComputation pipeline([=]() {
                cv::GMat in;

                cv::GMat frame = cv::gapi::copy(in);
                cv::GProtoOutputArgs outs = GOut(frame);

                cv::GMat detections = cv::gapi::infer<Faces>(in);

                cv::GArray<cv::Rect> faces = PostProc::on(detections, in, FLAGS_t);
                outs += GOut(faces);

                cv::GArray<cv::GMat> ages;
                cv::GArray<cv::GMat> genders;
                if (age_gender_enable) {
                    std::tie(ages, genders) = cv::gapi::infer<AgeGender>(faces, in);
                    outs += GOut(ages, genders);
                }

                cv::GArray<cv::GMat> y_fc;
                cv::GArray<cv::GMat> p_fc;
                cv::GArray<cv::GMat> r_fc;
                if (headpose_enable) {
                    std::tie(y_fc, p_fc, r_fc) = cv::gapi::infer<HeadPose>(faces, in);
                    outs += GOut(y_fc, p_fc, r_fc);
                }

                cv::GArray<cv::GMat> emotions;
                if (emotions_enable) {
                    emotions = cv::gapi::infer<Emotions>(faces, in);
                    outs += GOut(emotions);
                }

                cv::GArray<cv::GMat> landmarks;
                if (landmarks_enable) {
                    landmarks = cv::gapi::infer<FacialLandmark>(faces, in);
                    outs += GOut(landmarks);
                }

                return cv::GComputation(cv::GIn(in), std::move(outs));
        });


        std::string face_det_m = FLAGS_m;
        std::string face_det_w = fileNameNoExt(FLAGS_m) + ".bin";
        std::string face_det_d = FLAGS_d;
        auto det_net = cv::gapi::ie::Params<Faces> { face_det_m, face_det_w, face_det_d };

        std::string age_gen_det_m = FLAGS_m_ag;
        std::string age_gen_det_w = fileNameNoExt(FLAGS_m_ag) + ".bin";
        std::string age_gen_det_d = FLAGS_d_ag;
        auto age_net = cv::gapi::ie::Params<AgeGender> { age_gen_det_m, age_gen_det_w, age_gen_det_d }
                                                            .cfgOutputLayers({ "age_conv3", "prob" });

        std::string head_pose_det_m = FLAGS_m_hp;
        std::string head_pose_det_w = fileNameNoExt(FLAGS_m_hp) + ".bin";
        std::string head_pose_det_d = FLAGS_d_hp;
        auto hp_net = cv::gapi::ie::Params<HeadPose> { head_pose_det_m, head_pose_det_w, head_pose_det_d }
                                                        .cfgOutputLayers({ "angle_y_fc", "angle_p_fc", "angle_r_fc" });

        std::string landmarks_det_m = FLAGS_m_lm;
        std::string landmarks_det_w = fileNameNoExt(FLAGS_m_lm) + ".bin";
        std::string landmarks_det_d = FLAGS_d_lm;
        auto lm_net = cv::gapi::ie::Params<FacialLandmark> { landmarks_det_m, landmarks_det_w, landmarks_det_d };

        std::string emo_det_m = FLAGS_m_em;
        std::string emo_det_w = fileNameNoExt(FLAGS_m_em) + ".bin";
        std::string emo_det_d = FLAGS_d_em;
        auto emo_net = cv::gapi::ie::Params<Emotions> {  emo_det_m, emo_det_w, emo_det_d };

        // Form a kernel package (with a single OpenCV-based implementation of our
        // post-processing) and a network package (holding our three networks).x
        auto kernels = cv::gapi::kernels<OCVPostProc>();
        auto networks = cv::gapi::networks(det_net, age_net, hp_net, lm_net, emo_net);

        cv::GStreamingCompiled stream = pipeline.compileStreaming(cv::compile_args(kernels, networks));

        cv::Mat frame;
        std::vector<cv::Rect> face_hub;
        std::vector<cv::Mat> out_ages, out_genders;
        std::vector<cv::Mat> out_y_fc, out_p_fc, out_r_fc;
        std::vector<cv::Mat> out_landmarks;
        std::vector<cv::Mat> out_emotions;

        stream.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(FLAGS_i));

        cv::GRunArgsP out_vector;
        out_vector += cv::gout(frame);
        out_vector += cv::gout(face_hub);
        if (age_gender_enable) out_vector += cv::gout(out_ages, out_genders);
        if (headpose_enable)   out_vector += cv::gout(out_y_fc, out_p_fc, out_r_fc);
        if (emotions_enable)   out_vector += cv::gout(out_emotions);
        if (landmarks_enable)  out_vector += cv::gout(out_landmarks);

        Visualizer::Ptr visualizer;
        if (!FLAGS_no_show) {
            cv::namedWindow("Detection results");
            visualizer = std::make_shared<Visualizer>();
        } else {
            std::cout<< "To close the application, press 'CTRL+C' here" << std::endl; 
        }

        cv::VideoWriter videoWriter;

        Avg avg;

        stream.start();
        avg.start();
        while (stream.pull(std::move(out_vector)))
        {
            if (!FLAGS_no_show && emotions_enable && !FLAGS_no_show_emotion_bar) {
                visualizer->enableEmotionBar(frame.size(), {"neutral",
                                                            "happy",
                                                            "sad",
                                                            "surprise",
                                                            "anger"});
            }

            //  Postprocessing
            std::list<Face::Ptr> prev_faces;

            if (!FLAGS_no_smooth) {
                prev_faces.insert(prev_faces.begin(), faces.begin(), faces.end());
            }

            faces.clear();

            // For every detected face
            for (size_t i = 0; i < face_hub.size(); i++) {
                cv::Rect rect = face_hub[i] & cv::Rect({0, 0}, frame.size());

                Face::Ptr face;
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

                if (age_gender_enable) {
                    face->ageGenderEnable();
                    face->updateGender(out_genders[i].at<float>(0));
                    face->updateAge(out_ages[i].at<float>(0) * 100);
                }

                if (headpose_enable) {
                    face->headPoseEnable();
                    face->updateHeadPose({out_r_fc[i].at<float>(0),
                                          out_p_fc[i].at<float>(0),
                                          out_y_fc[i].at<float>(0)});
                }

                if (emotions_enable) {
                    face->emotionsEnable();
                    face->updateEmotions({
                                          {"neutral", out_emotions[i].at<float>(0)},
                                          {"happy", out_emotions[i].at<float>(1)} ,
                                          {"sad", out_emotions[i].at<float>(2)} ,
                                          {"surprise", out_emotions[i].at<float>(3)},
                                          {"anger", out_emotions[i].at<float>(4)}
                                          });
                }

                if (landmarks_enable) {
                    face->landmarksEnable();
                    std::vector<float> normedLandmarks;
                    size_t n_lm = 70;
                    for (size_t i_lm = 0UL; i_lm < n_lm; ++i_lm) {
                        normedLandmarks.push_back(out_landmarks[i].at<float>(2 * i_lm));
                        normedLandmarks.push_back(out_landmarks[i].at<float>(2 * i_lm + 1));
                    }

                    face->updateLandmarks(normedLandmarks);
                }
                // End of face postprocesing

                faces.push_back(face);
            }

            //  Visualizing results
            if (!FLAGS_no_show) {
                out.str("");
                out << "Total image throughput: " << std::fixed << std::setprecision(2)
                    << avg.fps(framesCounter) << " fps";
                cv::putText(frame, out.str(), cv::Point2f(10, 45), cv::FONT_HERSHEY_TRIPLEX, 1.2,
                            cv::Scalar(255, 0, 0), 2);

                // drawing faces
                visualizer->draw(frame, faces);

                cv::imshow("Detection results", frame);

                if (cv::waitKey(1) >= 0) stream.stop();
            }

            if (!FLAGS_o.empty() && !videoWriter.isOpened()) {
                videoWriter.open(FLAGS_o, cv::VideoWriter::fourcc('I', 'Y', 'U', 'V'), 25, cv::Size(frame.size()));
            }
            if (!FLAGS_o.empty()) {
                videoWriter.write(prev_frame);
            }

            framesCounter++;
        }

        if (!FLAGS_o.empty()) {
            videoWriter.release();
        }

        slog::info << "Number of processed frames: " << framesCounter << slog::endl;
        slog::info << "Total image throughput: " << avg.fps(framesCounter) << " fps" << slog::endl;

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
