// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <string>

#include <map>
#include <vector>

#include <inference_engine.hpp>


#include <ie_iextension.h>
#include <ext_list.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
#include <samples/common.hpp>

namespace gaze_estimation {
class IEWrapper {
public:
    IEWrapper(InferenceEngine::Core& ie,
              const std::string& modelPath,
              const std::string& deviceName, size_t batchSize, size_t irNum);
    // For setting input blobs containing images
    void setInputBlob(const std::string& blobName,const std::vector<cv::Mat>& images,
                      InferenceEngine::InferRequest& ir, int firstIndex);
    // For setting input blobs containing vectors of data
    void setInputBlob(const std::string& blobName, const std::vector<float>& data);

    void fillBlobs(const std::string& blobName,
                   const std::vector<cv::Mat>& images);

    // Get output blob content as a vector given its name (if there are more than one output blob)
    void getOutputBlob(const std::string& blobName, std::vector<float>& output);
    // Get output blob content as a vector (if there is only one output blob)
    void getOutputBlob(std::vector<float>& output);
    
    void setBatchSize(size_t size);

    size_t getBatchSize();

    void printPerlayerPerformance() const;

    const std::map<std::string, std::vector<unsigned long>>& getIputBlobDimsInfo() const;
    const std::map<std::string, std::vector<unsigned long>>& getOutputBlobDimsInfo() const;

    void reshape(const std::map<std::string, std::vector<unsigned long>>& newBlobsDimsInfo);

    void resizeNetwork();

    void infer();
    void infer(size_t ID);

    void startAsync();
    void startAsync(size_t ID);

    template<class T>
    void SetCompletionCallback(const T& callBackToSet);

//protected:
    std::string modelPath;
    std::string deviceName;
    InferenceEngine::Core& ie;
    InferenceEngine::CNNNetReader netReader;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executableNetwork;
    std::map<std::string, std::vector<unsigned long>> inputBlobsDimsInfo;
    std::map<std::string, std::vector<unsigned long>> outputBlobsDimsInfo;

    //new
    std::vector<InferenceEngine::InferRequest> InferRequests;
    std::map<std::shared_ptr<InferenceEngine::InferRequest>, int> infReqToCurrImg;
    int batchSize;
    int irNum;
    
    void setExecPart();
};
}  // namespace gaze_estimation
