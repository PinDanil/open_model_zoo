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
        GridMat gridMat = GridMat();

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
        
        size_t batchSize = inputImgs.size();
        size_t curImg = 0;

        //std::cout<<batchSize<<std::endl;

        //ieWrapper.network.setBatchSize(1);//TODO:rm in fufure

        std::queue<cv::Mat> showMats;
        std::condition_variable condVar;
        std::mutex mutex;
        std::mutex signal;
        std::unique_lock<std::mutex> lock(signal);
        ieWrapper.request.SetCompletionCallback(
                [&]{
                    //set some staff
                    
                    mutex.lock();
                    showMats.push(inputImgs[curImg%batchSize]);
                    curImg++;
                    mutex.unlock();
                    std::cout<<'0'<<std::endl;
                    condVar.notify_one();
                    std::cout<<'1'<<std::endl;
                    //set new Mat to ie   
                    //ieWrapper.setInputBlob(netName, inputImgs.at(curImg%batchSize));
                    
                    ieWrapper.startAsync();
                });

        
        
        cv::namedWindow("main");
        cv::imshow("main", gridMat.getMat());
        cv::Mat tmpMat;
        
        ieWrapper.setInputBlob(netName, inputImgs[curImg%batchSize]);
        ieWrapper.startAsync();

        while(true){
            std::cout<<'2'<<std::endl;
            
            condVar.wait(lock, [&]{return !showMats.empty();});
            mutex.lock();
            std::cout<<'3'<<std::endl;
            tmpMat = showMats.front();
            showMats.pop();
            mutex.unlock();
            //lock.unlock();
            
            std::cout<<'4'<<std::endl;
            gridMat.update(tmpMat);
            std::cout<<'5'<<std::endl;
            cv::imshow("main", gridMat.getMat());
            std::cout<<'6'<<std::endl;
            char key = static_cast<char>(cv::waitKey(10));
            //resultsMarker.toggle(key);
            // Press 'Esc' to quit, 'f' to flip the video horizontally
            if (key == 27)
                break;
        }
        
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