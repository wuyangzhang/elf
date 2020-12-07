//
// Created by wuyang zhang on 7/5/20.
//

#include "ControlManager.h"


ControlManager::ControlManager() {

    // connect to remote server sockets
    std::vector<std::string> addrs;
    for (const std::string &s: this->config.SERVERS_ADDR) {
        addrs.push_back(s);
    }
    this->remote_servers = connect_sockets(addrs);

    // connect to the LRC server
    //this->lrc_server = connect_socket(this->config.LRC_SERVER_ADDR);

    // create worker threads
    {
        this->runSignal = new std::atomic<int>[config.TOTAL_SERVER_NUM];
        this->workers = new std::thread[config.TOTAL_SERVER_NUM];
        for (int i = 0; i < config.TOTAL_SERVER_NUM; i++) {
            runSignal[i] = -1;
            void (ControlManager::*func)(int);
            func = &ControlManager::offload;
            workers[i] = std::thread(func, this, i);
        }
    }

};

ControlManager::~ControlManager() {
    delete this->runSignal;
}

void ControlManager::singleOffload(cv::Mat& image){
    this->images->at(0) = image;
    runSignal[0] = 0;

    while (finishSignal != 1){}
    finishSignal = 0;
}

void ControlManager::distFramePartitions(std::vector<cv::Mat> *images) {

    this->images = images;

    // notify worker thread to offload images..
    for (int i = 0; i < config.TOTAL_SERVER_NUM; i++) {
        runSignal[i] = 0;
    }

    while (finishSignal != config.TOTAL_SERVER_NUM) {}
    finishSignal = 0;

    // todo: merge results.
}

void ControlManager::offload(int index) {
    while (true) {

        // spin lock to wait for the offloading signal
        while (runSignal[index] != 0) {}
        this->runSignal[index] = -1;

        std::vector<uchar> buf;
        imencode(this->images->at(index), buf);

        zmq::message_t request(buf.size());
        memcpy(request.data(), buf.data(), buf.size());
        std::cout << "request size " << request.size() << std::endl;

        remote_servers->at(index)->send(request, zmq::send_flags::none);
        zmq::message_t reply;
        assert (remote_servers->at(index)->recv(reply, zmq::recv_flags::none) != -1);

        std::cout << "recv respond from the server " << reply.str() << std::endl;
        this->finishSignal++;
    }
}

const std::vector<float> ControlManager::getProcessTime(){
    return processTime;
}

