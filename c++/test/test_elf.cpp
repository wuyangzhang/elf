//
// Created by wuyang zhang on 7/18/20.
//

//
// Created by wuyang zhang on 7/15/20.
//

#include "../network/Socket.h"
#include "../partitioning/PartitionManager.h"

#include <iostream>
#include <chrono>         // std::chrono::seconds

#include <thread>
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable

#include "../network/Socket.h"
#include "../dataset/VideoLoader.h"
#include "../run/Client.h"



void run_elf_server(std::string port) {
    std::cout << "to run the server!" << std::endl;
    zmq::socket_t *socket = bind_socket(port);
    std::cout << "Start running the server!" << std::endl;

    Config config = Config();

    for (int i = 0; i < 100; i++) {
        zmq::message_t recv;
        //  Wait for next request from client
        assert (socket->recv(recv, zmq::recv_flags::none) != -1);

        std::vector<uchar> buf(recv.size());
        memcpy(buf.data(), recv.data(), recv.size());
        cv::Mat img;
        imdecode(img, buf);
        std::cout << "server " << port << " decoded image size " << img.size() << std::endl;

        // sleep for simulating function processing.. micro seconds
        std::this_thread::sleep_for(std::chrono::milliseconds (500));

        zmq::message_t reply(4);
        memcpy(reply.data(), "done", 4);
        socket->send(reply, zmq::send_flags::none);
    }

    close_socket(socket);
}



void run_elf_client() {
    Client client;
    client.run();
}


void test_elf() {
    std::thread threads[4];

    threads[0] = std::thread(run_elf_server, "5050");
    threads[1] = std::thread(run_elf_server, "5051");
    threads[2] = std::thread(run_elf_server, "5052");
    threads[3] = std::thread(run_elf_client);

    for(int i = 0; i < 4; i++){
        threads[0].join();
        threads[1].join();
        threads[2].join();
        threads[3].join();
    }

}
