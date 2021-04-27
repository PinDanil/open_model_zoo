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

G_API_OP(transpose, <cv::GMat(cv::GMat)>, "custom.transpose") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in) {
        return in.withSize(cv::Size(in.size.height, in.size.width));
    }
};

GAPI_OCV_KERNEL(OCVTranspose, transpose) {
    static void run(const cv::Mat &in,
                    cv::Mat &out) {
        cv::transpose(in, out);
    }
};

G_API_OP(reshape, <cv::GMat(cv::GMat, int, int)>, "custom.reshape") {
    static cv::GMatDesc outMeta(const cv::GMatDesc& in, const int d, const int c) {
        return cv::GMatDesc(d, c, in.size);
    }
};

GAPI_OCV_KERNEL(OCVReshape, reshape) {
    static void run(const cv::Mat &in, int d, int c,
                    cv::Mat &out) {
        out = in.reshape(c, d);
    }
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
        bboxes.clear();
/*
        const auto &in_ssd_dims = in_ssd_result.size;
        std::cout << "DIMS: " << in_ssd_dims.dims() << std::endl;
        CV_Assert(in_ssd_dims.dims() == 2u);

        const int MAX_PROPOSALS = in_ssd_dims[2];
        const int OBJECT_SIZE   = in_ssd_dims[3];
        std::cout << "MPROP: " << MAX_PROPOSALS << std::endl;
        std::cout << "MPROP: " << OBJECT_SIZE << std::endl;
        CV_Assert(OBJECT_SIZE == 7);

        const cv::Size upscale = in_frame.size();
        const cv::Rect surface({0,0}, upscale);
*/

        const int OBJECT_SIZE   = 5;

        const float *data = in_ssd_result.ptr<float>();
        for (int i = 0; i < 100; i++) {
            const float x_min = data[i * OBJECT_SIZE + 0];
            const float y_min = data[i * OBJECT_SIZE + 1];
            const float x_max = data[i * OBJECT_SIZE + 2];
            const float y_max = data[i * OBJECT_SIZE + 3];
            const float conf  = data[i * OBJECT_SIZE + 4];

            std::cout<< x_min << ' '<< x_max << ' '<< y_min << ' ' << y_max << ' ' <<conf<<std::endl;
            bboxes.push_back(cv::Rect(x_min, y_min, 
                                      x_max - x_min,
                                      y_max - y_min));
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

        cv::Size input_sz(320, 320); // HARDCODED size of input image
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
        };

        auto kernels = cv::gapi::kernels<OCVBoundingBoxExtract>();
        auto networks = cv::gapi::networks(person_detection);

        cv::GStreamingCompiled stream = pipeline.compileStreaming(cv::compile_args(kernels, networks));

        cv::VideoWriter videoWriter;
        cv::Mat frame, detections;
        auto out_vector = cv::gout(detections, frame);

        setInput(stream, FLAGS_i);
        stream.start();
        while (stream.pull(std::move(out_vector))){
                // Got bboxes and frame

                std::cout<< detections << std::endl;

                const float *data = detections.ptr<float>();
                for (int i =0; i < 100; i++) {
                    const float x_min = data[i * 5 + 0];
                    const float y_min = data[i * 5 + 1];
                    const float x_max = data[i * 5 + 2];
                    const float y_max = data[i * 5 + 3];
                    const float conf  = data[i * 5 + 4];
                }
/*
                if (!videoWriter.isOpened()) {
                    videoWriter.open(FLAGS_o, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, cv::Size(frame.size()));
                }
                if (!FLAGS_o.empty()) {
                    videoWriter.write(frame);
                }
*/
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
