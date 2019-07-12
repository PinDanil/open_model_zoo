# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME imagenet_classification_demo
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS classification_sample_async.h grid_mat.hpp
              DEPENDENCIES format_reader
              OPENCV_DEPENDENCIES core highgui videoio)
