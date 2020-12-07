//
// Created by wuyang zhang on 7/4/20.
//

#ifndef ELF_SOCKET_H
#define ELF_SOCKET_H

#include <zmq.hpp>

// create client connections
//const std::unique_ptr<std::vector<std::unique_ptr<zmq::socket_t>>> connect_sockets(std::vector<std::string>);
//
//const std::unique_ptr<zmq::socket_t> connect_socket(std::string);
//
//// server binding
//const std::unique_ptr<zmq::socket_t> bind_socket(std::string port);
//
//void close_socket(std::unique_ptr<zmq::socket_t>);
//
//void close_socket( std::unique_ptr<std::vector<std::unique_ptr<zmq::socket_t>>>);


std::vector<zmq::socket_t*>* connect_sockets(std::vector<std::string>);

zmq::socket_t* connect_socket(std::string);

// server binding
zmq::socket_t* bind_socket(std::string port);

void close_socket(zmq::socket_t*);

void close_socket(std::vector<zmq::socket_t*>*);

#endif //ELF_SOCKET_H
