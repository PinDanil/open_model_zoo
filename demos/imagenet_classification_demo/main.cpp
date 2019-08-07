// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* @brief The entry point the Inference Engine sample application
* @file classification_sample_async/main.cpp
* @example classification_sample_async/main.cpp
*/
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
        //ieWrapper.setBatchSize(1);
        //size_t batchSize = ieWrapper.getBatchSize();
        //ieWrapper.resizeNetwork(batchSize);
        // IRequest, model and devce is set.

        std::vector<std::string> imageNames;
        parseInputFilesArguments(imageNames);

        //TODO: set precision and layout!

        InputsDataMap inputInfo(ieWrapper.network.getInputsInfo());
        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies with 1 input only");
        std::string inputBlobName = inputInfo.begin()->first;

        //read all imgs
        std::vector<cv::Mat> inputImgs = {};
        for (const auto & i : imageNames) {
            inputImgs.push_back(cv::imread(i));
        }
        
        bool quitFlag = false;

        //Out info measure
        int64 framesNum = 0;
        int64 sumTime = 0;
        int64 startTime = 0;
        double lastInferTime = 0;
        double lastShowTime = cv::getTickCount();

        size_t curImg = 0;
        std::queue<cv::Mat> showMats;
        
        std::condition_variable condVar;
        std::mutex mutex;
        ieWrapper.request.SetCompletionCallback(
                [&]{
                    if(!quitFlag) {                        
                        {
                            std::lock_guard<std::mutex> lock(mutex);
                            //for(size_t i = 0; i < batchSize; ++i)
                                showMats.push(inputImgs[(curImg/*+i*/)%inputImgs.size()]);//!!
                            
                            curImg=(curImg+1)%inputImgs.size();

                            sumTime += lastInferTime = cv::getTickCount() - startTime; // >:-/
                            framesNum++;
                        }
                        condVar.notify_one();  
                    
                        ieWrapper.setInputBlob(inputBlobName, inputImgs, curImg);//!!
                        //ieWrapper.resizeNetwork(batchSize);

                        startTime = cv::getTickCount();
                        ieWrapper.startAsync();
                    }
                });

        GridMat gridMat = GridMat(10, 15);
        cv::namedWindow("main");
        cv::imshow("main", gridMat.getMat());

        ieWrapper.setInputBlob(inputBlobName, inputImgs, curImg);//!!
        startTime = cv::getTickCount();
        ieWrapper.startAsync();

        lastShowTime = cv::getTickCount();

        cv::Mat tmpMat;
        while(true) {      
            if( (cv::getTickCount() - lastShowTime) / cv::getTickFrequency() >= 0.05){
                double currSPF; 
                double overallSPF; 
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    while(showMats.empty()){   
                        condVar.wait(lock);
                    }
                    gridMat.update(showMats);
                    currSPF = (lastInferTime / cv::getTickFrequency());
                    overallSPF = (sumTime / cv::getTickFrequency()) / framesNum;
                }
                gridMat.textUpdate(overallSPF, currSPF);// overallTime is not protected
                //std::cout<< "Ov FPS: "<< 1./overallSPF << " Cur FPS: "<< 1./currSPF<<std::endl;
                cv::imshow("main", gridMat.getMat());
                
                lastShowTime = cv::getTickCount();
                
                char key = static_cast<char>(cv::waitKey(1));
                //Press 'Esc' to quit
                if (key == 27){
                    quitFlag = true;
                    break;
                }
            }
        }
        
        cv::destroyWindow("main");
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
