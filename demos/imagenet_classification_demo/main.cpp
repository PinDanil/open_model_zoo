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
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** This vector stores paths to the processed images **/
        std::vector<std::string> imageNames;
        parseInputFilesArguments(imageNames);
        if (imageNames.empty()) throw std::logic_error("No suitable images were found");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Creating Inference Engine" << slog::endl;

        Core ie;

        gaze_estimation::IEWrapper ieWrapper(ie, FLAGS_d, FLAGS_m);

        /** Printing device version **/
        std::cout << ie.GetVersions(FLAGS_d) << std::endl;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo(ieWrapper.getInputsInfo());
        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies with 1 input only");

        auto inputInfoItem = *inputInfo.begin();

        /** Specifying the precision and layout of input data provided by the user.
         * This should be called before load of the network to the device **/
        inputInfoItem.second->setPrecision(Precision::U8);
        inputInfoItem.second->setLayout(Layout::NCHW);

        std::vector<std::string> validImageNames = {};
        //std::vector<Blob::Ptr> inputsImg = {}; //prepared Blobs to InferRequest
        std::vector<cv::Mat> inputsMat = {}; //vector of Mats to show
        std::queue<cv::Mat> toShow; //que of Mats to show
       // long int curImg = 0;

        for (const auto & i : imageNames) {
            cv::Mat img = cv::imread(i);

            if(!img.empty()){
                validImageNames.push_back(i);
                inputsMat.push_back(img);
                //inputsImg.push_back(wrapMat2Blob(img));
            }
        }
        if (validImageNames.empty()) throw std::logic_error("Valid input images were not found!");

                /** Setting batch size using image count **/
        network.setBatchSize(inputsMat.size());
        size_t batchSize = network.getBatchSize();
        slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork executable_network = ie.LoadNetwork(network, FLAGS_d);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        slog::info << "Create infer request" << slog::endl;
        InferRequest inferRequest = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Do inference ---------------------------------------------------------
        size_t numIterations = 10;
        
        std::condition_variable notEmpty;
        std::mutex queMutex;
        std::unique_lock<std::mutex> lock(queMutex);

        GridMat gridMat;
        cv::Mat tmpPtr;

        cv::namedWindow("main window");

        inferRequest.SetCompletionCallback(
                [&] {
                    queMutex.lock();
                    //toShow.push_back(inputsMat[(curImg % inputsMat.size())]);
                    //++curImg;
                    //queMutex.unlock();               
                    notEmpty.notify_one();

                    inferRequest.StartAsync();
                });

        /* Start async request for the first time */
        slog::info << "Start inference (" << numIterations << " asynchronous executions)" << slog::endl;
        inferRequest.StartAsync();

        while(cv::waitKey(10) != 27) {
            notEmpty.wait(lock, [&]{ return !toShow.empty(); });

            queMutex.lock();
            tmpPtr = toShow.front();
            toShow.pop();
            queMutex.unlock();

            gridMat.update(tmpPtr);
            cv::imshow("main window",gridMat.getMat());
        }

        cv::destroyWindow("main window");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Process output -------------------------------------------------------

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
