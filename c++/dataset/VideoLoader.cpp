//
// Created by wuyang zhang on 7/4/20.
//

#include "VideoLoader.h"
#include <filesystem>
#include <fstream>


VideoLoader::VideoLoader(Config config) {
    this->rootVideoDir = config.VIDEO_DIR;
    this->generateWhiteList();
    this->collectVideoFrames(this->rootVideoDir);
    this->TOTAL_FRAMES_NUM = this->framesPath.size();
}

bool VideoLoader::hasNext() {
    return this->currIndex < this->TOTAL_FRAMES_NUM;
}

void VideoLoader::next(cv::Mat& img) {
    std::string path = this->framesPath.at(this->currIndex);
    this->currIndex += 1;
    this->loadImage(path, img);
}

void VideoLoader::cur(cv::Mat& img) {
    std::string path = this->framesPath.at(this->currIndex);
    this->loadImage(path, img);
}

struct comparisonFuction {
    inline int index(const std::string &str) {
        std::vector<std::string> ans;
        tokenize(str, '_', ans);
        std::string str1 = ans.at(1);
        tokenize(str1, '.', ans);
        ans.pop_back();
        return std::stoi(ans.back());
    }

    inline bool operator()(const std::string &str1, const std::string &str2) {
        return index(str1) < index(str2);
    }
};

/**
    * Given a root of video dir, it iterates the directory on the whitelist
    * and records the file names of all the video frames
    *
    * @param videoRootDir: the root directory including all the video folders
    */
void VideoLoader::collectVideoFrames(std::string videoRootDir) {
    std::vector<std::string> videoDirs;
    listDirs(videoRootDir, videoDirs);
    for (std::string dir : videoDirs) {
        if (!this->onWhiteList(dir)) {
            continue;
        }
        dir = videoRootDir + "/" + dir;
        if (!std::filesystem::is_directory(dir)) {
            continue;
        }

        std::vector<std::string> videos;
        listFiles(dir, videos);
        for (std::string imgAddr : videos) {
            std::vector<std::string> out;
            tokenize(imgAddr, '.', out);
            std::string type = out.back();
            if (type.compare("jpg") != 0 and type.compare("png") != 0) {
                continue;
            }
            imgAddr = dir + "/" + imgAddr;
            this->framesPath.push_back(imgAddr);
        }
    }

    std::sort(this->framesPath.begin(), this->framesPath.end(), comparisonFuction{});
}


bool VideoLoader::onWhiteList(std::string dir) {
    return this->whiteList.size() <= 0 || this->whiteList.find(dir) != this->whiteList.end();
}

void VideoLoader::generateWhiteList() {
    std::string line;
    std::ifstream file(this->rootVideoDir + "/whitelist.txt");
    if (file.is_open()) {
        while (getline(file, line)) {
            this->whiteList.insert(line);
        }
        file.close();
    }
}

void VideoLoader::loadImage(std::string imgAddr, cv::Mat& img) {
    img = cv::imread(imgAddr);
}


void VideoLoader::printFramesPath() {
    for (auto path: this->framesPath) {
        std::cout << path << std::endl;
    }
}

int VideoLoader::getTotalFrameNum(){
    return this->TOTAL_FRAMES_NUM;
}
