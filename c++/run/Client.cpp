//
// Created by wuyang zhang on 7/4/20.
//

#include "Client.h"

Client::Client() = default;


void Client::run() {
    Config config;
    VideoLoader videoLoader(config);
    cv::Mat img;
    while (videoLoader.hasNext()) {
        videoLoader.next(img);
        elf.run(img);
    }
}