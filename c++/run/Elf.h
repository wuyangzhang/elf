//
// Created by wuyang zhang on 7/17/20.
//

#ifndef ELF_ELF_H
#define ELF_ELF_H

#include "../network/Socket.h"
#include "../app/ApplicationInterface.h"
#include "../control/ControlManager.h"
#include "../partitioning/PartitionManager.h"
#include "../prediction/PredictionManager.h"


class Elf {
public:
    Elf();
    void run(cv::Mat&);

private:
    ApplicationInterface* applicationInterface;
    PredictionManager* predictionManager;
    PartitionManager* partitionManager;
    ControlManager* controlManager;

    int curId = -1;
};


#endif //ELF_ELF_H
