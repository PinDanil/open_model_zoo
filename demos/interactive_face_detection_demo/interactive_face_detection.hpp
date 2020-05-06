// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

/// @brief Message for help argument
static const char help_message[] = "Print a usage message";

/// @brief Message for images argument
static const char input_video_message[] = "Required. Path to a video file (specify \"cam\" to work with camera).";

/// @brief Message for images argument
static const char output_video_message[] = "Optional. Path to an output video file.";

/// @brief message for model IR argument
static const char face_detection_model_message[] = "Required. Path to an .xml file with a trained Face Detection model.";
static const char age_gender_model_message[] = "Optional. Path to an .xml file with a trained Age/Gender Recognition model.";
static const char head_pose_model_message[] = "Optional. Path to an .xml file with a trained Head Pose Estimation model.";
static const char emotions_model_message[] = "Optional. Path to an .xml file with a trained Emotions Recognition model.";
static const char facial_landmarks_model_message[] = "Optional. Path to an .xml file with a trained Facial Landmarks Estimation model.";

/// @brief Message for assigning face detection calculation to device
static const char target_device_message[] = "Optional. Target device for Face Detection network (the list of available devices is shown below). " \
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"The demo will look for a suitable plugin for a specified device.";

/// @brief Message for assigning age/gender calculation to device
static const char target_device_message_ag[] = "Optional. Target device for Age/Gender Recognition network (the list of available devices is shown below). " \
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"The demo will look for a suitable plugin for a specified device.";

/// @brief Message for assigning head pose calculation to device
static const char target_device_message_hp[] = "Optional. Target device for Head Pose Estimation network (the list of available devices is shown below). " \
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"The demo will look for a suitable plugin for a specified device.";

/// @brief Message for assigning emotions calculation to device
static const char target_device_message_em[] = "Optional. Target device for Emotions Recognition network (the list of available devices is shown below). " \
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"The demo will look for a suitable plugin for a specified device.";

/// @brief Message for assigning Facial Landmarks Estimation network to device
static const char target_device_message_lm[] = "Optional. Target device for Facial Landmarks Estimation network " \
"(the list of available devices is shown below). Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"The demo will look for a suitable plugin for device specified.";

/// @brief Message for probability threshold argument
static const char thresh_output_message[] = "Optional. Probability threshold for detections";

/// @brief Message do not show processed video
static const char no_show_processed_video[] = "Optional. Do not show processed video.";

/// @brief Message for fps argument
static const char fps_output_message[] = "Optional. Maximum FPS for playing video";

/// @brief Message for smooth argument
static const char no_smooth_output_message[] = "Optional. Do not smooth person attributes";

/// @brief Message for smooth argument
static const char no_show_emotion_bar_message[] = "Optional. Do not show emotion bar";


/// \brief Define flag for showing help message<br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for set image file<br>
/// It is a required parameter
DEFINE_string(i, "", input_video_message);

/// \brief Define parameter for an output video file<br>
/// It is an optional parameter
DEFINE_string(o, "", output_video_message);

/// \brief Define parameter for Face Detection model file<br>
/// It is a required parameter
DEFINE_string(m, "", face_detection_model_message);

/// \brief Define parameter for Age Gender Recognition model file<br>
/// It is a optional parameter
DEFINE_string(m_ag, "", age_gender_model_message);

/// \brief Define parameter for Head Pose Estimation model file<br>
/// It is a optional parameter
DEFINE_string(m_hp, "", head_pose_model_message);

/// \brief Define parameter for Emotions Recognition model file<br>
/// It is a optional parameter
DEFINE_string(m_em, "", emotions_model_message);

/// \brief Define parameter for Facial Landmarks Estimation model file<br>
/// It is an optional parameter
DEFINE_string(m_lm, "", facial_landmarks_model_message);

/// \brief target device for Face Detection network<br>
DEFINE_string(d, "CPU", target_device_message);

/// \brief Define parameter for target device for Age/Gender Recognition network<br>
DEFINE_string(d_ag, "CPU", target_device_message_ag);

/// \brief Define parameter for target device for Head Pose Estimation network<br>
DEFINE_string(d_hp, "CPU", target_device_message_hp);

/// \brief Define parameter for target device for Emotions Recognition network<br>
DEFINE_string(d_em, "CPU", target_device_message_em);

/// \brief Define parameter for target device for Facial Landmarks Estimation network<br>
DEFINE_string(d_lm, "CPU", target_device_message_lm);

/// \brief Define a parameter for probability threshold for detections<br>
/// It is an optional parameter
DEFINE_double(t, 0.5, thresh_output_message);

/// \brief Define a flag to disable smoothing person attributes<br>
/// It is an optional parameter
DEFINE_bool(no_smooth, false, no_smooth_output_message);

/// \brief Define a flag to disable showing processed video<br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

/// \brief Define a flag to disable showing emotion bar<br>
/// It is an optional parameter
DEFINE_bool(no_show_emotion_bar, false, no_show_emotion_bar_message);

/**
* \brief This function shows a help message
*/

static void showUsage() {
    std::cout << std::endl;
    std::cout << "interactive_face_detection [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                         " << help_message << std::endl;
    std::cout << "    -i \"<path>\"                " << input_video_message << std::endl;
    std::cout << "    -o \"<path>\"                " << output_video_message << std::endl;
    std::cout << "    -m \"<path>\"                " << face_detection_model_message<< std::endl;
    std::cout << "    -w \"<path>\"                " << face_detection_model_message<< std::endl;
    std::cout << "    -m_ag \"<path>\"             " << age_gender_model_message << std::endl;
    std::cout << "    -w_ag \"<path>\"             " << age_gender_model_message << std::endl;
    std::cout << "    -m_hp \"<path>\"             " << head_pose_model_message << std::endl;
    std::cout << "    -w_hp \"<path>\"             " << head_pose_model_message << std::endl;
    std::cout << "    -m_em \"<path>\"             " << emotions_model_message << std::endl;
    std::cout << "    -w_em \"<path>\"             " << emotions_model_message << std::endl;
    std::cout << "    -m_lm \"<path>\"             " << facial_landmarks_model_message << std::endl;
    std::cout << "    -w_lm \"<path>\"             " << facial_landmarks_model_message << std::endl;
    std::cout << "    -d \"<device>\"              " << target_device_message << std::endl;
    std::cout << "    -d_ag \"<device>\"           " << target_device_message_ag << std::endl;
    std::cout << "    -d_hp \"<device>\"           " << target_device_message_hp << std::endl;
    std::cout << "    -d_em \"<device>\"           " << target_device_message_em << std::endl;
    std::cout << "    -d_lm \"<device>\"           " << target_device_message_lm << std::endl;
    std::cout << "    -t                         " << thresh_output_message << std::endl;
    std::cout << "    -fps                       " << fps_output_message << std::endl;
    std::cout << "    -no_show                   " << no_show_processed_video << std::endl;
    std::cout << "    -no_smooth                 " << no_smooth_output_message << std::endl;
    std::cout << "    -no_show_emotion_bar       " << no_show_emotion_bar_message << std::endl;
}
