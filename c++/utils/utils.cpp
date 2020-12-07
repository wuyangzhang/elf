//
// Created by wuyang zhang on 7/11/20.
//

#include "utils.h"
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

void rescale(const cv::Mat &inImg, cv::Mat &outImg, float ratio) {
    cv::resize(inImg, outImg, cv::Size(), ratio, ratio);
}

void tokenize(std::string const &str, const char delim,
              std::vector<std::string> &out) {
    size_t start;
    size_t end = 0;

    while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
        end = str.find(delim, start);
        out.push_back(str.substr(start, end - start));
    }
}

void listFiles(std::string rootDir, std::vector<std::string> &ans) {
    const fs::path pathToShow{rootDir};

    for (const auto &entry : fs::directory_iterator(pathToShow)) {
        const auto filenameStr = entry.path().filename().string();
        if (entry.is_regular_file()) {
            ans.push_back(filenameStr);
        }
    }
    //std::sort(ans.begin(), ans.end(), comparisonFuction{});
}

void listDirs(std::string rootDir, std::vector<std::string> &ans) {
    const fs::path pathToShow{rootDir};
    for (const auto &entry : fs::directory_iterator(pathToShow)) {
        const auto filenameStr = entry.path().filename().string();
        if (entry.is_directory()) {
            ans.push_back(filenameStr);
        }
    }
    //std::sort(ans.begin(), ans.end());
}

void renderRP(cv::Mat &img, RP r, const cv::Scalar &color) {
    if (0 < r.x1 < r.x2 < img.cols and 0 < r.y1 < r.y2 < img.rows) {
        cv::Rect rect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1);
        rectangle(img, rect, color);
    }
}

void renderRPs(cv::Mat &img, std::vector<RP> &rps, const cv::Scalar &color) {
    for (RP r: rps) {
        renderRP(img, r, color);
    }
}

void tensorToRP(const torch::Tensor &tensor, std::vector<RP> &rps) {
    for (int z = 0; z < tensor.sizes()[0]; ++z) {
        RP r;
        r.x1 = tensor.index({z, 0}).item().toInt();
        r.y1 = tensor.index({z, 1}).item().toInt();
        r.x2 = tensor.index({z, 2}).item().toInt();
        r.y2 = tensor.index({z, 3}).item().toInt();
        rps.push_back(r);
    }

}

void display(const cv::Mat &img) {
    cv::imshow("show", img);
    cv::waitKey(5000);
}

struct comparisonFuction {
    inline int index(const std::string &str) {
        std::vector<std::string> ans;
        tokenize(str, '_', ans);
        std::string str1 = ans.at(1);
        tokenize(str1, '.', ans);
        ans.pop_back();
        std::cout << ans.back() << std::endl;
        return std::stoi(ans.back());
    }

    inline bool operator()(const std::string &str1, const std::string &str2) {
        return index(str1) < index(str2);
    }
};

void loadTensors(std::string tensor_dir, std::vector<torch::Tensor> &tensors) {
    std::vector<std::string> files;
    listFiles(tensor_dir, files);

    std::sort(files.begin(), files.end(), comparisonFuction{});

    for (std::string file: files) {
        torch::Tensor tensor;
        loadTensorFromFile(tensor_dir + file, tensor);
        tensors.push_back(tensor);
    }
}

void loadTensorFromFile(std::string file_path, torch::Tensor &tensor) {
//    std::cout << "load tensor file " << file_path << std::endl;
    std::string line;
    std::ifstream file(file_path);
    std::vector<int> values;
    if (file.is_open()) {
        while (getline(file, line)) {
            std::vector<std::string> out;
            tokenize(line, ' ', out);
            for (auto value: out) {
                int val = (int) std::stof(value);
                values.push_back(val);
            }
        }
        file.close();
    }

    tensor = torch::from_blob(values.data(), {(int) values.size() / 4, 4}, torch::kInt32).clone();
}


void imencode(cv::Mat& img, std::vector<uchar>& buf){
    std::vector<int> param = std::vector<int>(2);
    param[0]=cv::IMWRITE_JPEG_QUALITY;
    param[1]=80;//default(95) 0-100
    cv::imencode(".jpg",img,buf ,param);
}

void imdecode(cv::Mat& out, std::vector<uchar>& buf){
    out = cv::imdecode(cv::Mat(buf), cv::IMREAD_COLOR);
}

