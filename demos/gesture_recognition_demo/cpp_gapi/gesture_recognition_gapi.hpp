// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

#include <utils/default_flags.hpp>

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message.";
static const char class_map_message[] = "Print a usage message.";
static const char detection_model_message[] = "Print a usage message.";
static const char action_model_message[] = "Print a usage message.";

DEFINE_bool(h, false, help_message);
DEFINE_string(c, "", help_message);
DEFINE_string(m_a, "", help_message);
DEFINE_string(m_d, "", help_message);

/**
* \brief This function shows a help message
*/

static void showUsage() {
    std::cout << std::endl;
    std::cout << "gaze_estimation_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                       " << help_message << std::endl;
}
