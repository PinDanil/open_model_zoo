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


int main(int argc, char *argv[]) {
    try {
//         std::cout << "InferenceEngine: " << printable(*GetInferenceEngineVersion()) << std::endl;

        // ------------------------------ Parsing and validating of input arguments --------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
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
