//
// Created by wuyang zhang on 7/17/20.
//

#include "Elf.h"

Elf::Elf() {

    this->applicationInterface = new ApplicationInterface();

    this->predictionManager = new PredictionManager();

    this->partitionManager = new PartitionManager();

    this->controlManager = new ControlManager();
}

void Elf::run(cv::Mat &image) {
    curId++;

    if (predictionManager->isActive()) {
        // todo: LRC

        this->predictionManager->predict();

        // frame partition
        //this->partitionManager->getEqualRPBoxes(image);

        // todo: modify the api to accept tensors
        std::vector<RP> rpV;
        //torch::Tensor rps = this->predictionManager->getPredictedRP();
        torch::Tensor rps = this->predictionManager->getGTRP(curId);
        tensorToRP(rps, rpV);

        this->partitionManager->findRPBoxes(image, this->controlManager->getProcessTime(),
                                            rpV);
        this->partitionManager->partitionFrame(image);
        // offload
        this->controlManager->distFramePartitions(this->partitionManager->getFramePartitions());

    } else {
        this->controlManager->singleOffload(image);
    }

    // only for the test purpose
    torch::Tensor rps = this->predictionManager->getGTRP(curId);
    this->predictionManager->addHistTensors(rps);
}