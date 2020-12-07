//
// Created by wuyang zhang on 7/5/20.
//

#ifndef ELF_CONTROLMANAGER_H
#define ELF_CONTROLMANAGER_H

#include <pthread.h>

#include "../network/Socket.h"
#include "../app/ApplicationInterface.h"
#include "../partitioning/PartitionManager.h"
#include "../prediction/PredictionManager.h"
#include "../config/Config.h"
#include "../utils/utils.h"

class ControlManager {
public:
    ControlManager();
    void distFramePartitions(std::vector<cv::Mat>*);
    ~ControlManager();
    void singleOffload(cv::Mat&);
    const std::vector<float> getProcessTime();

private:
    void offload(int index);

    Config config;
    std::vector<zmq::socket_t*>* remote_servers;
    zmq::socket_t* lrc_server;

    std::vector<cv::Mat>* images;

    std::atomic<int>* runSignal;
    std::atomic<int> finishSignal = {0};
    std::thread* workers;

    std::vector<float> processTime;

};


#endif //ELF_CONTROLMANAGER_H
