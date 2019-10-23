// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gflags/gflags.h>
#include <iostream>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <iomanip>

#include <inference_engine.hpp>
#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif

#include <format_reader_ptr.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include <vpu/vpu_tools_common.hpp>
#include <vpu/vpu_plugin_config.hpp>

#include "segmentation_demo.h"

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    slog::info << "Parsing input parameters" << slog::endl;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

static std::map<std::string, std::string> configure(const std::string& confFileName) {
    auto config = parseConfig(confFileName);

    return config;
}

/**
 * @brief The entry point for inference engine deconvolution demo application
 * @file segmentation_demo/main.cpp
 * @example segmentation_demo/main.cpp
 */
int main(int argc, char *argv[]) {
    try {
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** This vector stores paths to the processed images **/
        std::vector<std::string> images;
        parseInputFilesArguments(images);
        if (images.empty()) throw std::logic_error("No suitable images were found");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

#ifdef WITH_EXTENSIONS
        /** Loading default extensions **/
        if (FLAGS_d.find("CPU") != std::string::npos) {
            /**
             * cpu_extensions library is compiled from "extension" folder containing
             * custom MKLDNNPlugin layer implementations. These layers are not supported
             * by mkldnn, but they can be useful for inferring custom topologies.
            **/
            ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
        }
#endif

        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
            ie.AddExtension(extension_ptr, "CPU");
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }
        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
            slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        }

        /** Printing device version **/

        slog::info << "Device info" << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d);

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;

        CNNNetReader networkReader;
        /** Read network model **/
        networkReader.ReadNetwork(FLAGS_m);

        /** Extract model name and load weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        networkReader.ReadWeights(binFileName);
        CNNNetwork network = networkReader.getNetwork();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------

        // --------------------------- Prepare input blobs -----------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo(network.getInputsInfo());
        /** Stores all input blobs data **/
        BlobMap inputBlobs;

        if (inputInfo.size() != 1) throw std::logic_error("Demo supports topologies only with 1 input");
        auto inputInfoItem = *inputInfo.begin();

        /** Collect images data ptrs **/
        std::vector<std::shared_ptr<unsigned char>> imagesData;
        for (auto & i : images) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Getting image data **/
            std::shared_ptr<unsigned char> data(reader->getData(inputInfoItem.second->getTensorDesc().getDims()[3],
                                                                inputInfoItem.second->getTensorDesc().getDims()[2]));
            if (data.get() != nullptr) {
                imagesData.push_back(data);
            }
        }
        if (imagesData.empty()) throw std::logic_error("Valid input images were not found!");

        /** Setting batch size using image count **/
        network.setBatchSize(imagesData.size());
        slog::info << "Batch size is " << std::to_string(networkReader.getNetwork().getBatchSize()) << slog::endl;

        inputInfoItem.second->setPrecision(Precision::U8);

        // --------------------------- Prepare output blobs ----------------------------------------------------
        slog::info << "Preparing output blobs" << slog::endl;

        OutputsDataMap outputInfo(network.getOutputsInfo());
        // BlobMap outputBlobs;
        std::string firstOutputName;

        for (auto & item : outputInfo) {
            if (firstOutputName.empty()) {
                firstOutputName = item.first;
            }
            DataPtr outputData = item.second;
            if (!outputData) {
                throw std::logic_error("output data pointer is not valid");
            }

            item.second->setPrecision(Precision::FP32);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork executable_network = ie.LoadNetwork(network, FLAGS_d, configure(FLAGS_config));
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        slog::info << "Create infer request" << slog::endl;
        InferRequest infer_request = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------
        /** Iterate over all the input blobs **/
        /** Iterating over all input blobs **/
        for (const auto & item : inputInfo) {
            /** Creating input blob **/
            Blob::Ptr input = infer_request.GetBlob(item.first);

            /** Fill input tensor with images. First r channel, then g and b channels **/
            size_t num_channels = input->getTensorDesc().getDims()[1];
            size_t image_size = input->getTensorDesc().getDims()[3] * input->getTensorDesc().getDims()[2];

            auto data = input->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();

            /** Iterate over all input images **/
            for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
                /** Iterate over all pixel in image (r,g,b) **/
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_channels; ++ch) {
                        /**          [images stride + channels stride + pixel id ] all in bytes            **/
                        data[image_id * image_size * num_channels + ch * image_size + pid] = imagesData.at(image_id).get()[pid*num_channels + ch];
                    }
                }
            }
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference ---------------------------------------------------------
        slog::info << "Start inference" << slog::endl;
        infer_request.Infer();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;

        const Blob::Ptr output_blob = infer_request.GetBlob(firstOutputName);
        const auto output_data = output_blob->buffer().as<float*>();

        size_t N = output_blob->getTensorDesc().getDims().at(0);
        size_t C, H, W;

        size_t output_blob_shape_size = output_blob->getTensorDesc().getDims().size();
        slog::info << "Output blob has " << output_blob_shape_size << " dimensions" << slog::endl;

        if (output_blob_shape_size == 3) {
            C = 1;
            H = output_blob->getTensorDesc().getDims().at(1);
            W = output_blob->getTensorDesc().getDims().at(2);
        } else if (output_blob_shape_size == 4) {
            C = output_blob->getTensorDesc().getDims().at(1);
            H = output_blob->getTensorDesc().getDims().at(2);
            W = output_blob->getTensorDesc().getDims().at(3);
        } else {
            throw std::logic_error("Unexpected output blob shape. Only 4D and 3D output blobs are supported.");
        }

        size_t image_stride = W*H*C;

        /** Iterating over all images **/
        for (size_t image = 0; image < N; ++image) {
            /** This vector stores pixels classes **/
            std::vector<std::vector<size_t>> outArrayClasses(H, std::vector<size_t>(W, 0));
            std::vector<std::vector<float>> outArrayProb(H, std::vector<float>(W, 0.));
            /** Iterating over each pixel **/
            for (size_t w = 0; w < W; ++w) {
                for (size_t h = 0; h < H; ++h) {
                    /* number of channels = 1 means that the output is already ArgMax'ed */
                    if (C == 1) {
                        outArrayClasses[h][w] = static_cast<size_t>(output_data[image_stride * image + W * h + w]);
                    } else {
                        /** Iterating over each class probability **/
                        for (size_t ch = 0; ch < C; ++ch) {
                            auto data = output_data[image_stride * image + W * H * ch + W * h + w];
                            if (data > outArrayProb[h][w]) {
                                outArrayClasses[h][w] = ch;
                                outArrayProb[h][w] = data;
                            }
                        }
                    }
                }
            }
            /** Dump resulting image **/
            std::string fileName = "out_" + std::to_string(image) + ".bmp";
            std::ofstream outFile(fileName, std::ofstream::binary);
            if (!outFile.is_open()) {
                throw std::logic_error("Can't open file : " + fileName);
            }

            writeOutputBmp(outArrayClasses, C, outFile);
            slog::info << "File : " << fileName << " was created" << slog::endl;
        }
        // -----------------------------------------------------------------------------------------------------
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
    slog::info << slog::endl << "This demo is an API example, for any performance measurements "
                                "please use the dedicated benchmark_app tool from the openVINO toolkit" << slog::endl;
    return 0;
}
