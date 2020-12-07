//
// Created by wuyang zhang on 7/12/20.
//

#ifndef ELF_PARTITIONSIMULATOR_H
#define ELF_PARTITIONSIMULATOR_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>


class PartitionSimulator {
public:
    void run();

private:
    void generateRandomImage(cv::Mat &);

};


#endif //ELF_PARTITIONSIMULATOR_H
