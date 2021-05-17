// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the gesture recognition
* \file gesture_recognition_gapi/main.cpp
* \example gesture_recognition_gapi/main.cpp
*/

#include <iostream>
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

G_TYPED_KERNEL(PersonTrack, <cv::GOpaque<cv::Rect>(cv::GArray<cv::Rect>)>, "custom.track") {
    static cv::GOpaqueDesc outMeta(const cv::GArrayDesc&) {
        return cv::empty_gopaque_desc();
    }
};

GAPI_OCV_KERNEL_ST(OCVPersonTrack, PersonTrack, std::vector<int>) {
 static void setup(const cv::GArrayDesc&,
                   std::shared_ptr<std::map<size_t, cv::Rect>> &tracked) {
        std::map<size_t, cv::Rect> persons = {};
        tracked = std::make_shared<std::map<size_t, cv::Rect>>(persons);
    }

    static void run(const std::vector<cv::Rect>& new_persons,
                    cv::Rect& out_person,
                    std::map<size_t, cv::Rect> &tracked) {}
};

G_API_OP(BoundingBoxExtract, <cv::GArray<cv::Rect>(cv::GMat, cv::GMat)>, "custom.bb_extract") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &in, const cv::GMatDesc &) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVBoundingBoxExtract, BoundingBoxExtract) {
    static void run(const cv::Mat &in_ssd_result,
                    const cv::Mat &in_frame,
                    std::vector<cv::Rect> &bboxes) {
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

            if (conf > 0.5){
                cv::Rect boundingBox(
                    static_cast<int>(x_min * scaling_x),
                    static_cast<int>(y_min * scaling_y),
                    static_cast<int>((x_max - x_min) * scaling_x),
                    static_cast<int>((y_max - y_min) * scaling_y)
                );

                bboxes.push_back(cv::Rect(static_cast<int>(x_min * scaling_x),
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

                cv::GMat detections = cv::gapi::infer<PersoneDetection>(in);
                
                cv::GMat out_frame = cv::gapi::copy(in);
                return cv::GComputation(cv::GIn(in), cv::GOut(detections, out_frame));
        });

        auto person_detection = cv::gapi::ie::Params<PersoneDetection> {
            FLAGS_m_d,                         // path to model
            fileNameNoExt(FLAGS_m_d) + ".bin", // path to weights
            "CPU"                              // device to use
        }.cfgOutputLayers({"boxes"});

        auto kernels = cv::gapi::kernels<OCVBoundingBoxExtract, OCVPersonTrack>();
        auto networks = cv::gapi::networks(person_detection);

        cv::VideoWriter videoWriter;
        cv::Mat detections, frame;
        auto out_vector = cv::gout(detections, frame);

        std::vector<cv::Rect> bbDetections;

        auto stream = pipeline.compileStreaming(cv::compile_args(kernels, networks));
        setInput(stream, FLAGS_i);
        stream.start();
        while (stream.pull(std::move(out_vector))){
                const float TRACKER_SCORE_THRESHOLD = 0.4;
                const float TRACKER_IOU_THRESHOLD = 0.3;



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
