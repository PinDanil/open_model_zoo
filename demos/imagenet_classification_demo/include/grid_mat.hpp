// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <set>
#include <string>
#include <vector>
#include <queue>

#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>

class GridMat {
public:
    cv::Mat outimg;

    explicit GridMat(size_t cNum, size_t rNum, size_t rh = 60,
    const cv::Size maxDisp = cv::Size{1080, 1920}):
    currSourceID{0}, rectangleHeight{rh}, positionNum{cNum * rNum} {
        cellSize.width = maxDisp.width * 1. / cNum;
        cellSize.height = (maxDisp.height - rectangleHeight) * 1. / rNum;
        
        //size_t nGridCols = static_cast<size_t>(ceil(sqrt(static_cast<float>(sizes.size()))));
        //size_t nGridRows = (sizes.size() - 1) / nGridCols + 1;

        for (size_t i = 0; i < cNum * rNum; i++) {
            cv::Point p;
            p.x = cellSize.width * (i % cNum);
            p.y = rectangleHeight + (cellSize.height * (i / cNum));
            points.push_back(p);
        }

        outimg.create((cellSize.height * rNum) + rectangleHeight, cellSize.width * cNum, CV_8UC3);
        outimg.setTo(0);
        clear();
    }

    cv::Size getCellSize() {
        return cellSize;
    }
/*
    void fill(std::vector<cv::Mat>& frames) {
        if (frames.size() > points.size()) {
            throw std::logic_error("Cannot display " + std::to_string(frames.size()) + " channels in a grid with " + std::to_string(points.size()) + " cells");
        }
        currSourceID = 0;
        for (size_t i = 0; i < frames.size(); i++) {
            cv::Mat cell = outimg(cv::Rect(points[i].x, points[i].y, cellSize.width, cellSize.height));

            if ((cellSize.width == frames[i].cols) && (cellSize.height == frames[i].rows)) {
                frames[i].copyTo(cell);
            } else if ((cellSize.width > frames[i].cols) && (cellSize.height > frames[i].rows)) {
                frames[i].copyTo(cell(cv::Rect(0, 0, frames[i].cols, frames[i].rows)));
            } else {
                cv::resize(frames[i], cell, cellSize);
            }
            currSourceID++;
        }
    }
*/
    void update(std::queue<cv::Mat>& frames) {
        while(!frames.empty()) {    
            cv::Mat cell = outimg(cv::Rect(points[currSourceID], cellSize));
            cv::Mat frame = frames.front();
            frames.pop();

            if ((cellSize.width == frame.cols) && (cellSize.height == frame.rows)) {
                frame.copyTo(cell);
            } /*else if ((cellSize.width > frame.cols) && (cellSize.height > frame.rows)) {
                frame.copyTo(cell(cv::Rect(0, 0, frame.cols, frame.rows)));
            }*/ else {
                cv::resize(frame, cell, cellSize);
            }
            
            if(currSourceID == points.size() - 1)
                currSourceID = 0;
            else
                currSourceID++;
        }
    }

    void textUpdate(double overSPF, double curSPF){
        //set rectangle 
        size_t colunmNum = outimg.cols;
        cv::Point p1 = cv::Point(0,0);
        cv::Point p2 = cv::Point(colunmNum, rectangleHeight);
        
        rectangle(outimg,p1,p2,
            cv::Scalar(0,0,0), cv::FILLED);
        
        //set text        
        auto frameWidth = outimg.cols;
        double fontScale = frameWidth * 1. / 640;
        auto fontColor = cv::Scalar(0, 255, 0);
        int thickness = 2;

        cv::putText(outimg,
                    cv::format("Overall FPS: %0.01f Current FPS: %0.01f Overall mSPF: %0.01f Current mSPF: %0.01f",
                    1./overSPF, 1./curSPF, 1000 * overSPF, 1000 * curSPF),
                    cv::Point(10, static_cast<int>(30 * fontScale / 1.6)),
                    cv::FONT_HERSHEY_PLAIN, fontScale, fontColor, thickness);
    }

    bool isFilled() const noexcept {
        return unupdatedSourceIDs.empty();
    }
    void clear() {
        size_t counter = 0;
        std::generate_n(std::inserter(unupdatedSourceIDs, unupdatedSourceIDs.end()), points.size(), [&counter]{return counter++;});
    }
    std::set<size_t> getUnupdatedSourceIDs() const noexcept {
        return unupdatedSourceIDs;
    }
    cv::Mat getMat() const noexcept {
        return outimg;
    }

private:
    cv::Size cellSize;
    size_t currSourceID;
    std::set<size_t> unupdatedSourceIDs;
    std::vector<cv::Point> points;
    size_t rectangleHeight;
    size_t positionNum;
};
