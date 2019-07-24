// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* @brief The entry point the Inference Engine sample application
* @file classification_sample_async/main.cpp
* @example classification_sample_async/main.cpp
*/
#ifndef __IMAGENET_CLASSIFICATION_DEMO__
#define __IMAGENET_CLASSIFICATION_DEMO__

#include <fstream>
#include <vector>
#include <queue>
#include <memory>
#include <string>
#include <map>
#include <condition_variable>
#include <mutex>

#include <inference_engine.hpp>

#include <format_reader_ptr.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <samples/classification_results.h>
#include <samples/ocv_common.hpp>
#include <samples/common.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <sys/stat.h>
#include <ext_list.hpp>

#include "classification_sample_async.h"
#include "ie_wrapper.hpp"
#include "grid_mat.hpp"

using namespace InferenceEngine;

ConsoleErrorListener error_listener;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    slog::info << "Parsing input parameters" << slog::endl;

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

int main(int argc, char *argv[]) {
    try {
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        Core ie;
        gaze_estimation::IEWrapper ieWrapper(ie, FLAGS_m, FLAGS_d);
        // IRequest, model and devce is set.

        std::vector<std::string> imageNames;
        parseInputFilesArguments(imageNames);

        //TODO: set precision and layout!

        InputsDataMap inputInfo(ieWrapper.network.getInputsInfo());
        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies with 1 input only");
        std::string netName = inputInfo.begin()->first;

        //read all imgs
        std::vector<cv::Mat> inputImgs = {};
        for (const auto & i : imageNames) {
            inputImgs.push_back(cv::imread(i));
        }
        
        bool quitFlag = false;

        int tInferenceBegins;
        int tInferenceEnds;
        double overallTime = 0.;

        size_t curImg = 0;
        size_t batchSize = inputImgs.size();
        std::queue<cv::Mat> showMats;
        
        std::condition_variable condVar;
        std::mutex mutex;
        ieWrapper.request.SetCompletionCallback(
                [&]{
                    if(!quitFlag) {
                    tInferenceEnds = cv::getTickCount();
                    overallTime = (tInferenceEnds - tInferenceBegins) * 1000. / cv::getTickFrequency();
                    
                    mutex.lock();
                    showMats.push(inputImgs[curImg%batchSize]);
                    curImg++;
                    mutex.unlock();
                    
                    condVar.notify_one();  
                    
                    ieWrapper.setInputBlob(netName, inputImgs.at(curImg%batchSize));
                    tInferenceBegins = cv::getTickCount();
                    ieWrapper.startAsync();
                    }
                });

        GridMat gridMat = GridMat();
        cv::namedWindow("main");
        cv::imshow("main", gridMat.getMat());

        ieWrapper.setInputBlob(netName, inputImgs[curImg%batchSize]);
        tInferenceBegins = cv::getTickCount();
        ieWrapper.startAsync();

        cv::Mat tmpMat;
        while(true){
            {
                std::unique_lock<std::mutex> lock(mutex);
                while(showMats.empty()){   
                condVar.wait(lock);
                }
                tmpMat = showMats.front();
                showMats.pop();
            }
            
            gridMat.update(tmpMat);
            gridMat.textUpdate(overallTime);
            cv::imshow("main", gridMat.getMat());
            
            char key = static_cast<char>(cv::waitKey(10));
            // Press 'Esc' to quit
            if (key == 27){
                quitFlag = true;
                break;
            }
        }
        
        cv::destroyWindow("main");

        return 0;
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
    slog::info << slog::endl << "This sample is an API example, for any performance measurements "
                                "please use the dedicated benchmark_app tool" << slog::endl;
    return 0;
}

#endif //__IMAGENET_CLASSIFICATION_DEMO__