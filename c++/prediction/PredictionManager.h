//
// Created by wuyang zhang on 7/5/20.
//

#ifndef ELF_PREDICTIONMANAGER_H
#define ELF_PREDICTIONMANAGER_H

#include <torch/torch.h>

#include "../utils/utils.h"
#include "../config/Config.h"

class PredictionManager {
public:
    PredictionManager();
    void predict();
    bool isActive();
    void addHistTensors(torch::Tensor);

    void printModelInfo();
    torch::Tensor getPredictedRP();

    torch::Tensor getGTRP(int index);

private:
    void loadModel(std::string);
    void moveModelToCuda(std::string);

    void preProcess(torch::Tensor & tensor);
    void postProcess(torch::Tensor & tensor);
    void rpIndex(torch::Tensor &input);
    void extendRP(torch::Tensor &tensor);

    const Config config;
    const int MAX_QUEUE_SIZE = config.WINDOW_SIZE;

    std::string modelPath;
    torch::jit::Module model;

    std::deque<torch::Tensor> histRPs;
    torch::Tensor predictedRP;
    const float RP_RESCALE_RATIO = config.RP_RESCALE_RATIO;

    std::vector<torch::Tensor> gtTensors;

};


#endif //ELF_PREDICTIONMANAGER_H
