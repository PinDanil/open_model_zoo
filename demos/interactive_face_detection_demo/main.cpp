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

        Timer timer;
        // read input (video) frame

        cv::Mat frame;

        const size_t width  = static_cast<size_t>(frame.cols);
        const size_t height = static_cast<size_t>(frame.rows);

        // ---------------------------------------------------------------------------------------------------
        // --------------------------- 1. Loading Inference Engine -----------------------------

        std::set<std::string> loadedDevices;
        std::vector<std::pair<std::string, std::string>> cmdOptions = {
            {FLAGS_d, FLAGS_m},
            {FLAGS_d_ag, FLAGS_m_ag},
            {FLAGS_d_hp, FLAGS_m_hp},
            {FLAGS_d_em, FLAGS_m_em},
            {FLAGS_d_lm, FLAGS_m_lm}
        };
        FaceDetection faceDetector(FLAGS_m, FLAGS_d, 1, false, FLAGS_async, FLAGS_t, FLAGS_r,
                                   static_cast<float>(FLAGS_bb_enlarge_coef), static_cast<float>(FLAGS_dx_coef), static_cast<float>(FLAGS_dy_coef));
        AgeGenderDetection ageGenderDetector(FLAGS_m_ag, FLAGS_d_ag, FLAGS_n_ag, FLAGS_dyn_ag, FLAGS_async, FLAGS_r);
        HeadPoseDetection headPoseDetector(FLAGS_m_hp, FLAGS_d_hp, FLAGS_n_hp, FLAGS_dyn_hp, FLAGS_async, FLAGS_r);
        EmotionsDetection emotionsDetector(FLAGS_m_em, FLAGS_d_em, FLAGS_n_em, FLAGS_dyn_em, FLAGS_async, FLAGS_r);
        FacialLandmarksDetection facialLandmarksDetector(FLAGS_m_lm, FLAGS_d_lm, FLAGS_n_lm, FLAGS_dyn_lm, FLAGS_async, FLAGS_r);

        /** Per-layer metrics **/

        // ---------------------------------------------------------------------------------------------------

        // --------------------------- 2. Reading IR models and loading them to plugins ----------------------
        // Disable dynamic batching for face detector as it processes one image at a time

        // ----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Doing inference -----------------------------------------------------
        // Starting inference & calculating performance
        slog::info << "Start inference " << slog::endl;

        // bool isFaceAnalyticsEnabled = ageGenderDetector.enabled() || headPoseDetector.enabled() ||
        //                               emotionsDetector.enabled() || facialLandmarksDetector.enabled();

        std::ostringstream out;
        size_t framesCounter = 0;
//        bool frameReadStatus;
//        bool isLastFrame;
        int delay = 1;
        double msrate = -1;
        cv::Mat prev_frame, next_frame;
        std::list<Face::Ptr> faces;
        size_t id = 0;

        if (FLAGS_fps > 0) {
            msrate = 1000.f / FLAGS_fps;
        }

        Visualizer::Ptr visualizer;
        if (!FLAGS_no_show || !FLAGS_o.empty()) {
            visualizer = std::make_shared<Visualizer>(cv::Size(width, height));
            if (!FLAGS_no_show_emotion_bar /*&& emotionsDetector.enabled()*/) {
                visualizer->enableEmotionBar(emotionsDetector.emotionsVec);
            }
        }

        // Detecting all faces on the first frame and reading the next one

        // Reading the next frame

        std::cout << "To close the application, press 'CTRL+C' here";
        if (!FLAGS_no_show) {
            std::cout << " or switch to the output window and press any key";
        }
        std::cout << std::endl;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ G-API STUFF START ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Describe networks we use in our program.
// In G-API, topologies act like "operations". Here we define our
// topologies as operations which have inputs and outputs.

// Every network requires three parameters to define:
// 1) Network's TYPE name - this TYPE is then used as a template
//    parameter to generic functions like cv::gapi::infer<>(),
//    and is used to define network's configuration (per-backend).
// 2) Network's SIGNATURE - a std::function<>-like record which defines
//    networks' input and output parameters (its API)
// 3) Network's IDENTIFIER - a string defining what the network is.
//    Must be unique within the pipeline.

// Note: these definitions are neutral to _how_ the networks are
// executed. The _how_ is defined at graph compilation stage (via parameters),
// not on the graph construction stage.

// Face detector: takes one Mat, returns another Mat
G_API_NET(Faces, <cv::GMat(cv::GMat)>, "face-detector");

// Age/Gender recognition - takes one Mat, returns two:
// one for Age and one for Gender. In G-API, multiple-return-value operations
// are defined using std::tuple<>.
using AGInfo = std::tuple<cv::GMat, cv::GMat>;
G_API_NET(AgeGender, <AGInfo(cv::GMat)>,   "age-gender-recoginition");

// Head pose recognition - takes one Mat, returns another.
using HPInfo = std::tuple<cv::GMat, cv::GMat, cv::GMat>;
G_API_NET(HeadPose, <HPInfo(cv::GMat)>,   "head-pose-recoginition");

// Facial landmark recognition - takes one Mat, returns another.
G_API_NET(FacialLandmark, <cv::GMat(cv::GMat)>,   "facial-landmark-recoginition");

// Emotion recognition - takes one Mat, returns another.
G_API_NET(Emotions, <cv::GMat(cv::GMat)>, "emotions-recognition");

// The kernel body is declared separately, this is just an interface.
// This operation takes two Mats (detections and the source image),
// and returns a vector of ROI (filtered by a default threshold).
// Threshold (or a class to select) may become a parameter, but since
// this kernel is custom, it doesn't make a lot of sense.
G_API_OP(PostProc, <cv::GArray<cv::Rect>(cv::GMat, cv::GMat)>, "custom.fd_postproc") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GMatDesc &) {
        // This function is required for G-API engine to figure out
        // what the output format is, given the input parameters.
        // Since the output is an array (with a specific type),
        // there's nothing to describe.
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
            // Declare an empty GMat - the beginning of the pipeline.
            cv::GMat in;

            // Run face detection on the input frame. Result is a single GMat,
            // internally representing an 1x1x200x7 SSD output.
            // This is a single-patch version of infer:
            // - Inference is running on the whole input image;
            // - Image is converted and resized to the network's expected format
            //   automatically.
            cv::GMat detections = cv::gapi::infer<Faces>(in);

            // Parse SSD output to a list of ROI (rectangles) using
            // a custom kernel. Note: parsing SSD may become a "standard" kernel.
            cv::GArray<cv::Rect> faces = PostProc::on(detections, in);

            // Now run Age/Gender model on every detected face. This model has two
            // outputs (for age and gender respectively).
            // A special ROI-list-oriented form of infer<>() is used here:
            // - First input argument is the list of rectangles to process,
            // - Second one is the image where to take ROI from;
            // - Crop/Resize/Layout conversion happens automatically for every image patch
            //   from the list
            // - Inference results are also returned in form of list (GArray<>)
            // - Since there're two outputs, infer<> return two arrays (via std::tuple).
            cv::GArray<cv::GMat> ages;
            cv::GArray<cv::GMat> genders;
            std::tie(ages, genders) = cv::gapi::infer<AgeGender>(faces, in);

            // Recognize axis–angle representation a on every face.
            // ROI-list-oriented infer<>() is used here as well.
            // Inference results are returned in form of list (GArray<>).
            // HeadPose network produce a three outputs (yaw, pitch and roll).
            // Since there're three outputs, infer<> return three arrays (via std::tuple).
            cv::GArray<cv::GMat> y_fc;
            cv::GArray<cv::GMat> p_fc;
            cv::GArray<cv::GMat> r_fc;
            std::tie(y_fc, p_fc, r_fc) = cv::gapi::infer<HeadPose>(faces, in);

            // Recognize landmarks on every face.
            // ROI-list-oriented infer<>() is used here as well.
            // Since FacialLandmark network produce a single output, only one
            // GArray<> is returned here.
            cv::GArray<cv::GMat> landmarks = cv::gapi::infer<FacialLandmark>(faces, in);

            // Recognize emotions on every face.
            // ROI-list-oriented infer<>() is used here as well.
            // Since Emotions network produce a single output, only one
            // GArray<> is returned here.
            cv::GArray<cv::GMat> emotions = cv::gapi::infer<Emotions>(faces, in);

            // Return the decoded frame as a result as well.
            // Input matrix can't be specified as output one, so use copy() here
            // (this copy will be optimized out in the future).
            cv::GMat frame = cv::gapi::copy(in);

            // Now specify the computation's boundaries - our pipeline consumes
            // one images and produces five outputs.
            return cv::GComputation(cv::GIn(in),
                                    cv::GOut(frame, detections, ages, genders, 
                                             y_fc, p_fc, r_fc,
                                             emotions,
                                             landmarks));
        });

    // Note: it might be very useful to have dimensions loaded at this point!
    // After our computation is defined, specify how it should be executed.
    // Execution is defined by inference backends and kernel backends we use to
    // compile the pipeline (it is a different step).

    // Declare IE parameters for FaceDetection network. Note here Face
    // is the type name we specified in GAPI_NETWORK() previously.
    // cv::gapi::ie::Params<> is a generic configuration description which is
    // specialized to every particular network we use.
    //
    // OpenCV DNN backend will have its own parmater structure with settings
    // relevant to OpenCV DNN module. Same applies to other possible inference
    // backends, like cuDNN, etc (:-))
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

//    auto out_vector = cv::gout(imgBeautif, imgShow, vctFaceConts,
//                               vctElsConts, vctRects);

//    cv::Mat out_frame;
    cv::Mat out_detections;
    std::vector<cv::Rect> face_hub;
    std::vector<cv::Mat> out_ages;
    std::vector<cv::Mat> out_genders;
    std::vector<cv::Mat> out_y_fc, out_p_fc, out_r_fc; 
    std::vector<cv::Mat> out_landmarks;
    std::vector<cv::Mat> out_emotions;
    
    stream.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(FLAGS_i));

    cv::GRunArgsP out_vector = cv::gout(frame, out_detections, out_ages, out_genders,
                                        out_y_fc, out_p_fc, out_r_fc,
                                        out_emotions, out_landmarks);
    cv::namedWindow("Detection results");

    stream.start();
    while (stream.running())
    {
        timer.start("total");
        
        stream.pull(std::move(out_vector));

        if (!FLAGS_no_show && -1 != cv::waitKey(delay)) break;

        faceDetector.fetchResults(out_detections,
                                  static_cast<float>(frame.cols),
                                  static_cast<float>(frame.rows));
        ageGenderDetector.fetchResults(out_ages, out_genders);
        headPoseDetector.fetchResults(out_y_fc, out_p_fc, out_r_fc);
        emotionsDetector.fetchResults(out_emotions);
        facialLandmarksDetector.fetchResults(out_landmarks);

        auto prev_detection_results = faceDetector.results;
        
        //  Postprocessing
        std::list<Face::Ptr> prev_faces;

        if (!FLAGS_no_smooth) {
            prev_faces.insert(prev_faces.begin(), faces.begin(), faces.end());
        }

        faces.clear();
    
        // For every detected face
        for (size_t i = 0; i < prev_detection_results.size(); i++) {
            auto& result = prev_detection_results[i];
            cv::Rect rect = result.location & cv::Rect({0, 0}, frame.size());
            
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

            face->ageGenderEnable(/*(ageGenderDetector.enabled() &&
                                   i < ageGenderDetector.maxBatch)*/
                                   true);
            if (/*face->isAgeGenderEnabled()*/ true) {
                AgeGenderDetection::Result ageGenderResult = ageGenderDetector[i];
                face->updateGender(ageGenderResult.maleProb);
                face->updateAge(ageGenderResult.age);
            }

            face->headPoseEnable(/*(headPoseDetector.enabled() &&
                                  i < headPoseDetector.maxBatch)*/true);
            if (/*face->isHeadPoseEnabled()*/ true) {
                face->updateHeadPose(headPoseDetector[i]);
            }

            face->emotionsEnable(/*(emotionsDetector.enabled() &&
                                  i < emotionsDetector.maxBatch)*/ true);
            if (/*face->isEmotionsEnabled()*/ true) {
                face->updateEmotions(emotionsDetector[i]);
            }

            face->landmarksEnable(/*(facialLandmarksDetector.enabled() &&
                                   i < facialLandmarksDetector.maxBatch)*/ true);
            if (/*face->isLandmarksEnabled()*/ true) {
                face->updateLandmarks(facialLandmarksDetector[i]);
            }

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  G-API STUFF END  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        slog::info << "Number of processed frames: " << framesCounter << slog::endl;
        slog::info << "Total image throughput: " << framesCounter * (1000.f / timer["total"].getTotalDuration()) << " fps" << slog::endl;

        // Showing performance results
        if (FLAGS_pc) {
            // faceDetector.printPerformanceCounts(getFullDeviceName(ie, FLAGS_d));
            // ageGenderDetector.printPerformanceCounts(getFullDeviceName(ie, FLAGS_d_ag));
            // headPoseDetector.printPerformanceCounts(getFullDeviceName(ie, FLAGS_d_hp));
            // emotionsDetector.printPerformanceCounts(getFullDeviceName(ie, FLAGS_d_em));
            // facialLandmarksDetector.printPerformanceCounts(getFullDeviceName(ie, FLAGS_d_lm));
        }
        // ---------------------------------------------------------------------------------------------------

        if (!FLAGS_o.empty()) {
            // videoWriter.release();
        }

        // release input video stream
        // cap.release();

        // close windows
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
