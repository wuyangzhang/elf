//
// Created by wuyang zhang on 7/5/20.
//
#include <iostream>
#include <pthread.h>

#include "../network/Socket.h"


void *run_server(void *) {
    std::cout << "to run the server!" << std::endl;
    zmq::socket_t *socket = bind_socket("5050");
    std::cout << "Start running the server!" << std::endl;

    for (int i = 0; i < 10; i++) {
        zmq::message_t request;
        //  Wait for next request from client
        //socket->recv(&request, 0);
        assert (socket->recv(request, zmq::recv_flags::none) != -1);

        std::cout << "Receive the message " << request.to_string() << " from the client" << std::endl;
        zmq::message_t reply(5);
        memcpy(reply.data(), "World", 5);
        socket->send(reply, zmq::send_flags::none);
    }

    close_socket(socket);
    return NULL;
}


void *run_client(void *) {
    std::cout << "to run the client!" << std::endl;
    zmq::socket_t *socket = connect_socket("127.0.0.1:5050");
    std::cout << "[Client] connected to the server!" << std::endl;
    for (int i = 0; i < 10; i++) {
        zmq::message_t request(5);
        memcpy(request.data(), "hello", 5);
        socket->send(request, zmq::send_flags::none);

        zmq::message_t reply;
        assert (socket->recv(reply, zmq::recv_flags::none) != -1);

        std::cout << "Receive the message " << reply.to_string() << " from the server" << std::endl;
    }
    close_socket(socket);
    return NULL;
}

void test_network() {
    pthread_t threads[2];

    if (pthread_create(&threads[1], NULL, run_server, NULL)) {
        std::cout << "Error:unable to create server thread" << std::endl;
        exit(-1);
    }

    if (pthread_create(&threads[0], NULL, run_client, NULL)) {
        std::cout << "Error:unable to create client thread" << std::endl;
        exit(-1);
    }

    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);

    std::cout << "thread finished" << std::endl;

}

