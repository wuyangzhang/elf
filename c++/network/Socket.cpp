//
// Created by wuyang zhang on 7/4/20.
//

#include "Socket.h"

#include <iostream>


std::vector<zmq::socket_t *> *connect_sockets(std::vector<std::string> addrs) {
    std::vector<zmq::socket_t *> *sockets = new std::vector<zmq::socket_t *>;
    for (std::string addr: addrs) {
        zmq::socket_t *socket = connect_socket(addr);
        sockets->push_back(socket);
    }
    return sockets;
}

zmq::socket_t *connect_socket(std::string addr) {
    zmq::context_t *context = new zmq::context_t(1);
    //zmq::context_t context(1);
    zmq::socket_t *socket = new zmq::socket_t(*context, ZMQ_REQ);
    std::cout << "connect to the socket " << "tcp://" + addr << " c" << std::endl;
    socket->connect("tcp://" + addr);
    return socket;
}

// server binding
zmq::socket_t *bind_socket(std::string port) {
    zmq::context_t *context = new zmq::context_t(1);
    zmq::socket_t *socket = new zmq::socket_t(*context, ZMQ_REP);
    std::cout << "bind to the socket " << "tcp://*:" + port << std::endl;
    socket->bind("tcp://*:" + port);
    return socket;
}

void close_socket(zmq::socket_t *socket) {
    socket->close();
    delete socket;
}

void close_socket(std::vector<zmq::socket_t *> *sockets) {
    for (auto it = sockets->begin(); it != sockets->end(); it++) {
        close_socket(*it);
    }
}


//const std::unique_ptr<std::vector<std::unique_ptr<zmq::socket_t>>> connect_sockets(std::vector<std::string> addrs){
//    std::unique_ptr<std::vector<std::unique_ptr<zmq::socket_t>>> sockets(new std::vector<zmq::socket_t>());
//    for(std::string addr: addrs){
//        std::unique_ptr<zmq::socket_t> socket = connect_socket(addr);
//        sockets->push_back(socket);
//    }
//    return sockets;
//}
//
//const std::unique_ptr<zmq::socket_t> connect_socket(std::string addr){
//    std::unique_ptr<zmq::context_t> context(new zmq::context_t(1));
//    std::unique_ptr<zmq::socket_t> socket(new zmq::socket_t(*context, ZMQ_REQ));
//
//    //socket.connect ("tcp://localhost:5555");
//    socket->connect ("tcp://" + addr);
//    return std::move(socket);
//}
//
//const std::unique_ptr<zmq::socket_t> bind_socket(std::string port){
//
//    //std::unique_ptr<Song> song2(new Song(L"Nothing on You", L"Bruno Mars"));
////    std::unique_ptr<zmq::context_t> context(new zmq::context_t(1));
////    std::unique_ptr<zmq::socket_t> socket(new zmq::socket_t(*context, ZMQ_REQ));
//
//    zmq::context_t* context = new zmq::context_t(1);
//    zmq::socket_t* socket = new zmq::socket_t(*context, ZMQ_REP);
//    //socket->bind("tcp://*:" + port);
//
//    socket->bind("tcp://*:5050");
//    return socket;
//}
//
//void close_sockets(std::unique_ptr<std::vector<std::unique_ptr<zmq::socket_t>>> sockets){
//    for(auto it = sockets->begin(); it != sockets->end(); it++){
//        close_socket(*it);
//    }
//}
//
//void close_socket(zmq::socket_t* socket){
//    socket->close();
//}



