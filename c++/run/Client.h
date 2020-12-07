//
// Created by wuyang zhang on 7/4/20.
//

#ifndef ELF_CLIENT_H
#define ELF_CLIENT_H

#include <iostream>
#include <chrono>
#include <zmq.hpp>


#include "../dataset/VideoLoader.h"
#include "../config/Config.h"
#include "Elf.h"


class Client {
public:
    Client();
    void run();

private:
    Config config;
    Elf elf;
};


#endif //ELF_CLIENT_H
