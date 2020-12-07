//
// Created by wuyang zhang on 7/11/20.
//

#ifndef ELF_UTILS_H
#define ELF_UTILS_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

void tokenize(std::string const &str, const char delim,
              std::vector<std::string> &out);

void listFiles(std::string rootDir, std::vector<std::string> &ans);

void listDirs(std::string rootDir, std::vector<std::string> &ans);

struct RP {
    int x1;
    int y1;
    int x2;
    int y2;

    RP() {
        this->x1 = this->y1 = this->x2 = this->y2 = 0;
    }

    RP(const RP &r): x1(r.x1), y1(r.y1), x2(r.x2), y2(r.y2) {
    }

    RP(const int a, const int b, const int c, const int d): x1(a), y1(b), x2(c), y2(d){
    }
};

void rescale(const cv::Mat &, cv::Mat &, float ratio);

void renderRPs(cv::Mat &, std::vector<RP> &, const cv::Scalar &);

void renderRP(cv::Mat &, RP, const cv::Scalar &);

void tensorToRP(const torch::Tensor &, std::vector<RP> &);

void display(const cv::Mat &);

void imencode(cv::Mat&, std::vector<uchar>&);

void imdecode(cv::Mat&, std::vector<uchar>&);

void loadTensors(std::string, std::vector<torch::Tensor> &);

void loadTensorFromFile(std::string, torch::Tensor &);

#endif //ELF_UTILS_H
