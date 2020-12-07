//
// Created by wuyang zhang on 7/12/20.
//

#include "PartitionSimulator.h"
#include <chrono>

#include "../utils/utils.h"

#include "../partitioning/PartitionManager.h"

void PartitionSimulator::run() {
    std::vector<torch::Tensor> tensors;
    loadTensors("/Users/wuyang/PycharmProjects/ParSim/raw_txt/", tensors);

    PartitionManager partitionManager;
    for (auto t: tensors) {
        cv::Mat img;
        generateRandomImage(img);
        std::vector<RP> rps;
        tensorToRP(t, rps);
        renderRPs(img, rps, cv::Scalar(255, 0, 0));

        std::vector<float> latency;

        auto t1 = std::chrono::high_resolution_clock::now();
        partitionManager.findRPBoxes(img, latency, rps);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
        std::cout << "running time: " << duration << std::endl;

//        std::cout << "RP box coordinates" << std::endl;
//        for (auto r: *partitionManager.getRPBoxCoords()) {
//            std::cout << r.x1 << " " << r.x2 << " " << r.y1 << " " << r.y2 << std::endl;
//        }
        renderRPs(img, *partitionManager.getRPBoxCoords(), cv::Scalar(0, 255, 255));
        //display(img);
    }
}

void PartitionSimulator::generateRandomImage(cv::Mat &image) {
    image = cv::Mat(375, 1242, CV_8UC3, cv::Scalar(0, 255, 0));
}


