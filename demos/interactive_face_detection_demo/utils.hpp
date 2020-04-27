// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

# pragma once

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

#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>

#include <ie_iextension.h>
#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif

#include <opencv2/opencv.hpp>

// -------------------------Generic routines for detection networks-------------------------------------------------

struct FaceDetectionResult {
    int label;
    float confidence;
    cv::Rect location;
};

struct AgeGenderResult {
    float age;
    float maleProb;
};

struct HeadPoseResults {
    float angle_r;
    float angle_p;
    float angle_y;
};


class CallStat {
public:
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

    CallStat();

    double getSmoothedDuration();
    double getTotalDuration();
    double getLastCallDuration();
    void calculateDuration();
    void setStartTime();

private:
    size_t _number_of_calls;
    double _total_duration;
    double _last_call_duration;
    double _smoothed_duration;
    std::chrono::time_point<std::chrono::high_resolution_clock> _last_call_start;
};

class Timer {
public:
    void start(const std::string& name);
    void finish(const std::string& name);
    CallStat& operator[](const std::string& name);

private:
    std::map<std::string, CallStat> _timers;
};
