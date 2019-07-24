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

    explicit GridMat(const cv::Size maxDisp = cv::Size{1920, 1080}) {
        currSourceID = 0;
        
        size_t maxWidth = 54;
        size_t maxHeight = 64;

        size_t nGridCols = 20;
        size_t nGridRows = 30;
        size_t gridMaxWidth = static_cast<size_t>(maxDisp.width/nGridCols);
        size_t gridMaxHeight = static_cast<size_t>(maxDisp.height/nGridRows);

        float scaleWidth = static_cast<float>(gridMaxWidth) / maxWidth;
        float scaleHeight = static_cast<float>(gridMaxHeight) / maxHeight;
        float scaleFactor = std::min(1.f, std::min(scaleWidth, scaleHeight));

        cellSize.width = static_cast<int>(maxWidth * scaleFactor);
        cellSize.height = static_cast<int>(maxHeight * scaleFactor);

        for (size_t i = 0; i < 600; i++) { // 600 = 30*20
            cv::Point p;
            p.x = cellSize.width * (i % nGridCols);
            p.y = cellSize.height * (i / nGridCols);
            points.push_back(p);
        }

        outimg.create(cellSize.height * nGridRows, cellSize.width * nGridCols, CV_8UC3);
        outimg.setTo(0);
        clear();
    }

    cv::Size getCellSize() {
        return cellSize;
    }

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

    void update(const cv::Mat& frame) {
        cv::Mat cell = outimg(cv::Rect(points[currSourceID % points.size()], cellSize));

        if ((cellSize.width == frame.cols) && (cellSize.height == frame.rows)) {
            frame.copyTo(cell);
        } else if ((cellSize.width > frame.cols) && (cellSize.height > frame.rows)) {
            frame.copyTo(cell(cv::Rect(0, 0, frame.cols, frame.rows)));
        } else {
            cv::resize(frame, cell, cellSize);
        }

        currSourceID++;

        //textUpdate("Wow");
    }

    void textUpdate(double FPS){
        auto frameHeight = outimg.rows;
        double fontScale = 1.6 * frameHeight / 640;
        auto fontColor = cv::Scalar(0, 255, 0);
        int thickness = 2;

        cv::putText(outimg,
                    cv::format("Overall FPS: %0.0f", FPS),
                    cv::Point(10, static_cast<int>(30 * fontScale / 1.6)),
                    cv::FONT_HERSHEY_PLAIN, fontScale, fontColor, thickness);
        /*
        cv::putText(outimg, str,
                    cv::Point2f(10, 35),
                    cv::FONT_HERSHEY_PLAIN,
                    0.7, cv::Scalar{255, 255, 255});
        */
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
    //General frame size
    cv::Size cellSize;
    //Current pos in outing
    size_t currSourceID;
    std::set<size_t> unupdatedSourceIDs;
    std::vector<cv::Point> points;
};

void fillROIColor(cv::Mat& displayImage, cv::Rect roi, cv::Scalar color, double opacity) {
    if (opacity > 0) {
        roi = roi & cv::Rect(0, 0, displayImage.cols, displayImage.rows);
        cv::Mat textROI = displayImage(roi);
        cv::addWeighted(color, opacity, textROI, 1.0 - opacity , 0.0, textROI);
    }
}

void putTextOnImage(cv::Mat& displayImage, std::string str, cv::Point p,
                    cv::HersheyFonts font, double fontScale, cv::Scalar color,
                    int thickness = 1, cv::Scalar bgcolor = cv::Scalar(),
                    double opacity = 0) {
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(str, font, 0.5, 1, &baseline);
    fillROIColor(displayImage, cv::Rect(cv::Point(p.x, p.y + baseline),
                                        cv::Point(p.x + textSize.width, p.y - textSize.height)),
                 bgcolor, opacity);
    cv::putText(displayImage, str, p, font, fontScale, color, thickness);
}
