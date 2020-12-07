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


bool ready = false;
std::mutex mtx;
std::condition_variable conVar;
zmq::socket_t *sockets[3];
cv::Mat* images[3];

std::atomic<int> runSignal[3] = {-1, -1, -1};
std::atomic<int> finishSignal = 0;


void run_parallel_server(std::string port) {
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


void offloadThread(int index){
    while(true){
        while(runSignal[index] != 0){}
        runSignal[index] = -1;

        std::vector<uchar> buf;
        imencode(*images[index], buf);

        zmq::message_t request(buf.size());
        memcpy(request.data(), buf.data(), buf.size());
        std::cout << "request size " << request.size() << std::endl;

        sockets[index]->send(request, zmq::send_flags::none);
        zmq::message_t reply;
        assert (sockets[index]->recv(reply, zmq::recv_flags::none) != -1);

        std::cout << "recv respond from the server " << reply.str() << std::endl;
        finishSignal++;
    }

}


void run_parallel_client() {
    std::cout << "to run the client!" << std::endl;

    Config config = Config();
    VideoLoader videoLoader(config);
    cv::Mat img;

    // create multi-sockets
    sockets[0] = connect_socket("127.0.0.1:5050");
    sockets[1] = connect_socket("127.0.0.1:5051");
    sockets[2] = connect_socket("127.0.0.1:5052");

    std::cout << "[Client] connected to the server!" << std::endl;

    // create worker threads
    std::thread workers[3];
    for(int i = 0; i < 3; i++){
        workers[i] = std::thread(offloadThread, i);
    }

    std::cout << "[Client] created worker threads!" << std::endl;
    std::cout << "total image count " << videoLoader.getTotalFrameNum() << std::endl;
    int i =0 ;
    while (videoLoader.hasNext()) {
        //std::unique_lock<std::mutex> lck(mtx);
        auto start = std::chrono::high_resolution_clock::now();
        videoLoader.next(img);
        images[0] = images[1] = images[2] = &img;
        //lck.unlock();
        //ready = true;
//        std::lock_guard<std::mutex> lk(mtx);
//        conVar.notify_all();
        for (int j = 0; j < 3; j++) {
            runSignal[j] = 0;
        }

        // wait for workers
        while (finishSignal != 3) {}
        finishSignal = 0;

        auto end = std::chrono::high_resolution_clock::now();

        // Calculating total time taken by the program.
        double time_taken =
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "cur index " << i++ << " runs for " << time_taken << "ms" << std::endl;
//        std::unique_lock<std::mutex> lck(mtx);
//        conVar.wait(lck, []{return counter == 3;});
//        counter = 0;
    }

    close_socket(sockets[0]);
    close_socket(sockets[1]);
    close_socket(sockets[2]);

}


void test_parallel_offload() {
    std::thread threads[4];

    threads[0] = std::thread(run_parallel_server, "5050");
    threads[1] = std::thread(run_parallel_server, "5051");
    threads[2] = std::thread(run_parallel_server, "5052");
    threads[3] = std::thread(run_parallel_client);

    for(int i = 0; i < 4; i++){
        threads[0].join();
        threads[1].join();
        threads[2].join();
        threads[3].join();
    }

}
