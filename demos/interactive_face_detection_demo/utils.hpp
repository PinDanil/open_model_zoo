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

    CallStat():
        _number_of_calls(0), _total_duration(0.0), _last_call_duration(0.0), _smoothed_duration(-1.0) {
    }

    double getSmoothedDuration() {
        // Additional check is needed for the first frame while duration of the first
        // visualisation is not calculated yet.
        if (_smoothed_duration < 0) {
            auto t = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<ms>(t - _last_call_start).count();
        }
        return _smoothed_duration;
    }

    double getTotalDuration() {
        return _total_duration;
    }

    double getLastCallDuration() {
        return _last_call_duration;
    }

    void calculateDuration() {
        auto t = std::chrono::high_resolution_clock::now();
        _last_call_duration = std::chrono::duration_cast<ms>(t - _last_call_start).count();
        _number_of_calls++;
        _total_duration += _last_call_duration;
        if (_smoothed_duration < 0) {
            _smoothed_duration = _last_call_duration;
        }
        double alpha = 0.1;
        _smoothed_duration = _smoothed_duration * (1.0 - alpha) + _last_call_duration * alpha;
    }

    void setStartTime() {
        _last_call_start = std::chrono::high_resolution_clock::now();
    }

private:
    size_t _number_of_calls;
    double _total_duration;
    double _last_call_duration;
    double _smoothed_duration;
    std::chrono::time_point<std::chrono::high_resolution_clock> _last_call_start;
};

class Timer {
public:
    void start(const std::string& name) {
        if (_timers.find(name) == _timers.end()) {
            _timers[name] = CallStat();
        }
        _timers[name].setStartTime();
    }

    void finish(const std::string& name) {
        auto& timer = (*this)[name];
        timer.calculateDuration();
    }

    CallStat& operator[](const std::string& name) {
        if (_timers.find(name) == _timers.end()) {
            throw std::logic_error("No timer with name " + name + ".");
        }
        return _timers[name];
    }

private:
    std::map<std::string, CallStat> _timers;
};
