//
// Created by wuyang zhang on 7/11/20.
//

#include <iostream>
#include <pthread.h>

#include "../network/Socket.h"
#include "../dataset/VideoLoader.h"


void *run_img_server(void *) {
    std::cout << "to run the server!" << std::endl;
    zmq::socket_t *socket = bind_socket("5050");
    std::cout << "Start running the server!" << std::endl;

    Config config = Config();

    for (int i = 0; i < 10; i++) {
        zmq::message_t recv;
        //  Wait for next request from client
        //socket->recv(&request, 0);
        assert (socket->recv(recv, zmq::recv_flags::none) == 0);

        std::vector<uchar> buf(recv.size());
        memcpy(buf.data(), recv.data(), recv.size());

        cv::Mat img;
        imdecode(img, buf);

        std::cout << "decoded image size " << img.size() << std::endl;

        zmq::message_t reply(4);
        memcpy(reply.data(), "done", 4);
        socket->send(reply, zmq::send_flags::none);

    }

    close_socket(socket);
    return NULL;
}


void *run_img_client(void *) {
    std::cout << "to run the client!" << std::endl;
    zmq::socket_t *socket = connect_socket("127.0.0.1:5050");
    std::cout << "[Client] connected to the server!" << std::endl;

    Config config = Config();
    VideoLoader videoLoader(config);
    cv::Mat img;

    while (videoLoader.hasNext()) {
        videoLoader.next(img);
        std::vector<uchar> buf;
        imencode(img, buf);

        zmq::message_t request(buf.size());

        memcpy(request.data(), buf.data(), buf.size());
        std::cout << "request size " << request.size() << std::endl;

        socket->send(request, zmq::send_flags::none);
        zmq::message_t reply;
        assert (socket->recv(reply, zmq::recv_flags::none) == 0);
    }

    close_socket(socket);
    return NULL;
}


void test_img_transfer() {
    pthread_t threads[2];

    if (pthread_create(&threads[1], NULL, run_img_server, NULL)) {
        std::cout << "Error:unable to create server thread" << std::endl;
        exit(-1);
    }

    if (pthread_create(&threads[0], NULL, run_img_client, NULL)) {
        std::cout << "Error:unable to create client thread" << std::endl;
        exit(-1);
    }

    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);

    pthread_exit(NULL);
}