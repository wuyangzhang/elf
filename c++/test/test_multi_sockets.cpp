//
// Created by wuyang zhang on 7/16/20.
//

//
// Created by wuyang zhang on 7/15/20.
//

#include "../network/Socket.h"
#include "../partitioning/PartitionManager.h"

#include <iostream>
#include <unistd.h>

#include <thread>
#include <mutex>              // std::mutex, std::unique_lock

#include "../dataset/VideoLoader.h"


zmq::socket_t *multi_sockets[3];

void run_multi_socket_server(std::string port) {
    std::cout << "to run the server!" << std::endl;
    zmq::socket_t *socket = bind_socket(port);
    std::cout << "Start running the server!" << std::endl;

    Config config = Config();

    for (int i = 0; i < 100; i++) {
        zmq::message_t recv;
        //  Wait for next request from client
        assert (socket->recv(recv, zmq::recv_flags::none) != -1);
        std::cout << "Receive the message " << recv.to_string() << " from the client" << std::endl;

//        std::vector<uchar> buf(recv.size());
//        memcpy(buf.data(), recv.data(), recv.size());

        // sleep for simulating function processing.. micro seconds
        usleep(10 * 1000);

        zmq::message_t reply(4);
        memcpy(reply.data(), "done", 4);
        socket->send(reply, zmq::send_flags::none);
    }

    close_socket(socket);
}


void run_multi_socket_client() {
    std::cout << "to run the client!" << std::endl;

    Config config = Config();
    // create multi-sockets
    multi_sockets[0] = connect_socket("127.0.0.1:5050");
    multi_sockets[1] = connect_socket("127.0.0.1:5051");
    multi_sockets[2] = connect_socket("127.0.0.1:5052");


    std::cout << "[Client] connected to the server!" << std::endl;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 3; j++) {
            zmq::message_t request(5);
            memcpy(request.data(), "hello", 5);
            multi_sockets[j]->send(request, zmq::send_flags::none);

            zmq::message_t reply;
            assert (multi_sockets[j]->recv(reply, zmq::recv_flags::none) != -1);

            std::cout << "Receive the message " << reply.to_string() << " from the server" << std::endl;
        }

    }

    close_socket(multi_sockets[0]);
    close_socket(multi_sockets[1]);
    close_socket(multi_sockets[2]);

}


void test_multi_sockets() {
    std::thread threads[4];

    threads[0] = std::thread(run_multi_socket_server, "5050");
    threads[1] = std::thread(run_multi_socket_server, "5051");
    threads[2] = std::thread(run_multi_socket_server, "5052");
    threads[3] = std::thread(run_multi_socket_client);

    threads[0].join();
    threads[1].join();
    threads[2].join();
    threads[3].join();
}
