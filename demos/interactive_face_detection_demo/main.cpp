// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine interactive_face_detection demo application
* \file interactive_face_detection_demo/main.cpp
* \example interactive_face_detection_demo/main.cpp
*/
#include <vector>
#include <string>
#include <list>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

#include "interactive_face_detection.hpp"
#include "utils.hpp"
#include "face.hpp"
#include "visualizer.hpp"

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

using AGInfo = std::tuple<cv::GMat, cv::GMat>;
using HPInfo = std::tuple<cv::GMat, cv::GMat, cv::GMat>;
G_API_NET(Faces,          <cv::GMat(cv::GMat)>, "face-detector");
G_API_NET(AgeGender,      <AGInfo(cv::GMat)>,   "age-gender-recoginition");
G_API_NET(HeadPose,       <HPInfo(cv::GMat)>,   "head-pose-recoginition");
G_API_NET(FacialLandmark, <cv::GMat(cv::GMat)>, "facial-landmark-recoginition");
G_API_NET(Emotions,       <cv::GMat(cv::GMat)>, "emotions-recognition");

G_API_OP(PostProc, <cv::GArray<cv::Rect>(cv::GMat, cv::GMat, float, float, float, float)>, "custom.fd_postproc") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GMatDesc &, float, float, float, float) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVPostProc, PostProc) {
    static void run(const cv::Mat &in_ssd_result,
                    const cv::Mat &in_frame,
                    float threshold,
                    float bb_enlarge_coefficient,
                    float bb_dx_coefficient,
                    float bb_dy_coefficient,
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
            if (confidence < threshold) {
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

void rawOutputDetections(const int i,
                         const cv::Mat &ssd_result,
                         const cv::Size upscale,
                         const float detectionThreshold) {
    const auto &in_ssd_dims = ssd_result.size;
    CV_Assert(in_ssd_dims.dims() == 4u);

    const int OBJECT_SIZE   = in_ssd_dims[3];
    CV_Assert(OBJECT_SIZE == 7);

    const float *data = ssd_result.ptr<float>();

    const float image_id   = data[i * OBJECT_SIZE + 0]; // batch id
    const float label      = data[i * OBJECT_SIZE + 1];
    const float confidence = data[i * OBJECT_SIZE + 2];
    const float rc_left    = data[i * OBJECT_SIZE + 3];
    const float rc_top     = data[i * OBJECT_SIZE + 4];
    const float rc_right   = data[i * OBJECT_SIZE + 5];
    const float rc_bottom  = data[i * OBJECT_SIZE + 6];

    if (image_id >= 0) {
        int x      = static_cast<int>(rc_left   * upscale.width);
        int y      = static_cast<int>(rc_top    * upscale.height);
        int width  = static_cast<int>(rc_right  * upscale.width)  - x;
        int height = static_cast<int>(rc_bottom * upscale.height) - y;

        std::cout << "[" << i << "," << label << "] element, prob = " << confidence <<
             "    (" << x << "," << y << ")-(" << width << "," << height << ")"
             << ((confidence > detectionThreshold) ? " WILL BE RENDERED!" : "") << std::endl;
    }
}

void rawOutputAgeGender(const int idx, const cv::Mat out_ages, const cv::Mat out_genders) {
    const float *age_data = out_ages.ptr<float>();
    const float *gender_data = out_genders.ptr<float>();

    float maleProb = gender_data[1];
    float age      = age_data[0] * 100;

    std::cout << "[" << idx << "] element, male prob = " << maleProb << ", age = " << age << std::endl;        
}

void rawOutputHeadpose(const int idx,
                       const cv::Mat out_y_fc,
                       const cv::Mat out_p_fc,
                       const cv::Mat out_r_fc) {
    const float *y_data = out_y_fc.ptr<float>();
    const float *p_data = out_p_fc.ptr<float>();
    const float *r_data = out_r_fc.ptr<float>();

    std::cout << "[" << idx << "] element, yaw = " << y_data[0] <<
                 ", pitch = " << p_data[0] <<
                 ", roll = " << r_data[0]  << std::endl;
}

void rawOutputLandmarks(const int idx, const cv::Mat out_landmark) {
    const float *lm_data = out_landmark.ptr<float>();

    std::cout << "[" << idx << "] element, normed facial landmarks coordinates (x, y):" << std::endl;

    // FIXME: extract n_lm from out_landmarks[idx]
    int n_lm = 70;
    for (int i_lm = 0; i_lm < n_lm / 2; ++i_lm) {
        float normed_x = lm_data[2 * i_lm];
        float normed_y = lm_data[2 * i_lm + 1];

        std::cout << normed_x << ", " << normed_y << std::endl;
    }
}

void rawOutputEmotions(const int idx, const cv::Mat out_emotion) {
    static const std::vector<std::string> emotionsVec = {"neutral", "happy", "sad", "surprise", "anger"};
    size_t emotionsVecSize = emotionsVec.size();

    const float *em_data = out_emotion.ptr<float>();

    std::cout << "[" << idx << "] element, predicted emotions (name = prob):" << std::endl;
    for (size_t i = 0; i < emotionsVecSize; i++) {
        std::cout << emotionsVec[i] << " = " << em_data[i];
        if (emotionsVecSize - 1 != i) {
            std::cout << ", ";
        } else {
            std::cout << std::endl;
        }
    }
}

void faceProcessing(cv::Mat frame,
                    Face::Ptr &face,
                    cv::Rect face_rect,
                    std::list<Face::Ptr> &prev_faces,
                    std::vector<cv::Rect> &face_hub,
                    size_t &id,
                    bool no_smooth) {
    // Face apdate
    cv::Rect rect = face_rect & cv::Rect({0, 0}, frame.size());

    if (!no_smooth) {
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
}

void ageGenderProcessing(Face::Ptr &face,
                         cv::Mat out_age,
                         cv::Mat out_gender) {
    const float *age_data =    out_age.ptr<float>();
    const float *gender_data = out_gender.ptr<float>();

    float maleProb = gender_data[1];
    float age      = age_data[0] * 100;

    face->updateGender(maleProb);
    face->updateAge(age);
}

void headPoseProcessing(Face::Ptr &face,
                        cv::Mat out_y_fc,
                        cv::Mat out_p_fc,
                        cv::Mat out_r_fc) {
    const float *y_data = out_y_fc.ptr<float>();
    const float *p_data = out_p_fc.ptr<float>();
    const float *r_data = out_r_fc.ptr<float>();

    face->updateHeadPose({y_data[0],
                          p_data[0],
                          r_data[0]});
}

void emotionsProcessing(Face::Ptr &face, cv::Mat out_emotion) {
    const float *em_data = out_emotion.ptr<float>();

    face->updateEmotions({
                          {"neutral", em_data[0]},
                          {"happy", em_data[1]} ,
                          {"sad", em_data[2]} ,
                          {"surprise", em_data[3]},
                          {"anger", em_data[4]}
                          });
}

void landmarksProcessing(Face::Ptr &face, cv::Mat out_landmark) {
    const float *lm_data = out_landmark.ptr<float>();

    std::vector<float> normedLandmarks;
    size_t n_lm = 70;
    for (size_t i_lm = 0UL; i_lm < n_lm; ++i_lm) {
        normedLandmarks.push_back(lm_data[2 * i_lm]);
        normedLandmarks.push_back(lm_data[2 * i_lm + 1]);
    }

    face->updateLandmarks(normedLandmarks);
}

int main(int argc, char *argv[]) {
    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validating of input arguments --------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        if (FLAGS_n_ag != 0)
            std::cout << "[ WARNING ] Option \"-num_batch_ag\" is not supported in this version of the demo.\n";
        if (FLAGS_n_hp != 0)
            std::cout << "[ WARNING ] Option \"-num_batch_hp\" is not supported in this version of the demo.\n";
        if (FLAGS_n_em != 0)
            std::cout << "[ WARNING ] Option \"-num_batch_em\" is not supported in this version of the demo.\n";
        if (FLAGS_n_lm != 0)
            std::cout << "[ WARNING ] Option \"-num_batch_lm\" is not supported in this version of the demo.\n";
        if (FLAGS_dyn_ag != false)
            std::cout << "[ WARNING ] Option \"-dyn_batch_ag\" is not supported in this version of the demo.\n";
        if (FLAGS_dyn_hp != false)
            std::cout << "[ WARNING ] Option \"-dyn_batch_hp\" is not supported in this version of the demo.\n";
        if (FLAGS_dyn_em != false)
            std::cout << "[ WARNING ] Option \"-dyn_batch_em\" is not supported in this version of the demo.\n";
        if (FLAGS_dyn_lm != false)
            std::cout << "[ WARNING ] Option \"-dyn_batch_lm\" is not supported in this version of the demo.\n";
        if (FLAGS_pc != false)
            std::cout << "[ WARNING ] Option \"-pc\" is not supported in this version of the demo.\n";
        if (!FLAGS_c.empty())
            std::cout << "[ WARNING ] Option \"-c\" is not supported in this version of the demo.\n";
        if (!FLAGS_l.empty())
            std::cout << "[ WARNING ] Option \"-l\" is not supported in this version of the demo.\n";


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
                outs += GOut(detections);

                cv::GArray<cv::Rect> faces = PostProc::on(detections, in,
                                                          FLAGS_t,
                                                          FLAGS_bb_enlarge_coef,
                                                          FLAGS_dx_coef,
                                                          FLAGS_dy_coef);
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
        cv::Mat ssd_res;
        std::vector<cv::Rect> face_hub;
        std::vector<cv::Mat> out_ages, out_genders;
        std::vector<cv::Mat> out_y_fc, out_p_fc, out_r_fc;
        std::vector<cv::Mat> out_landmarks;
        std::vector<cv::Mat> out_emotions;

        stream.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(FLAGS_i));

        cv::GRunArgsP out_vector;
        out_vector += cv::gout(frame);
        out_vector += cv::gout(ssd_res);
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

        Timer timer;

        stream.start();

        while (stream.running())
        {
            timer.start("total");

            if (!stream.pull(std::move(out_vector))) {
                std::cout<<"End of streaming" << std::endl;
                if(FLAGS_loop_video) {
                    stream.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(FLAGS_i));
                    stream.start();
                } else if (!FLAGS_no_wait) {
                    std::cout << "No more frames to process!" << std::endl;
                    cv::waitKey(0);
                }
            } else {
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
                    Face::Ptr face;

                    cv::Rect rect = face_hub[i] & cv::Rect({0, 0}, frame.size());
                    faceProcessing(frame, face, rect,
                                   prev_faces, face_hub,
                                   id, FLAGS_no_smooth);
                    if(FLAGS_r)
                        rawOutputDetections(i, ssd_res, frame.size(), FLAGS_t);

                    if (age_gender_enable) {
                        ageGenderProcessing(face, out_ages[i], out_genders[i]);
                        if(FLAGS_r)
                            rawOutputAgeGender(i, out_ages[i], out_genders[i]);
                    }

                    if (headpose_enable) {
                        headPoseProcessing(face, out_y_fc[i], out_p_fc[i], out_r_fc[i]);
                        if(FLAGS_r)
                            rawOutputHeadpose(i, out_y_fc[i], out_p_fc[i], out_r_fc[i]);
                    }

                    if (emotions_enable) {
                        emotionsProcessing(face, out_emotions[i]);
                        if(FLAGS_r)
                            rawOutputEmotions(i, out_emotions[i]);
                    }

                    if (landmarks_enable) {
                        landmarksProcessing(face, out_landmarks[i]);
                        if(FLAGS_r)
                            rawOutputLandmarks(i, out_landmarks[i]);
                    }
                    // End of face postprocesing

                    faces.push_back(face);
                }

                //  Visualizing results
                if (!FLAGS_no_show) {
                    out.str("");
                    out << "Total image throughput: " << std::fixed << std::setprecision(2)
                        << 1000.f / (timer["total"].getSmoothedDuration()) << " fps";
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
                timer.finish("total");
            }
        }

        if (!FLAGS_o.empty()) {
            videoWriter.release();
        }

        slog::info << "Number of processed frames: " << framesCounter << slog::endl;
        slog::info << "Total image throughput: " << framesCounter * (1000.f / timer["total"].getTotalDuration()) << " fps" << slog::endl;

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
