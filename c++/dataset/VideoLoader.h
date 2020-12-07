//
// Created by wuyang zhang on 7/4/20.
//

#ifndef ELF_VIDEOLOADER_H
#define ELF_VIDEOLOADER_H

#include <iostream>
#include <unordered_set>
#include <vector>
#include <opencv2/opencv.hpp>

#include "../utils/utils.h"

#include "../config/Config.h"


class VideoLoader {
public:
    VideoLoader(Config config);
    bool hasNext();
    void next(cv::Mat&);
    void cur(cv::Mat&);
    void collectVideoFrames(std::string videoRootDir);

    bool onWhiteList(std::string dir);
    void generateWhiteList();
    void loadImage(std::string imgAddr,  cv::Mat& img);

    void printFramesPath();
    int getTotalFrameNum();

private:
    std::string rootVideoDir;
    std::unordered_set<std::string> whiteList;
    std::vector<std::string> framesPath;
    int TOTAL_FRAMES_NUM;
    int currIndex = 0;
};


#endif //ELF_VIDEOLOADER_H
