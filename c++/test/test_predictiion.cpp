//
// Created by wuyang zhang on 7/22/20.
//

#include "../prediction/PredictionManager.h"
#include "../partitioning/PartitionSimulator.h"

void test_prediction(){

    PartitionSimulator partitionSimulator;
    std::vector<torch::Tensor> tensors;
    loadTensors("/Users/wuyang/PycharmProjects/ParSim/raw_txt/", tensors);

    PredictionManager predictionManager;

    for (auto t: tensors) {
        predictionManager.addHistTensors(t);
        if(predictionManager.isActive()){
            predictionManager.predict();
        }
    }
}