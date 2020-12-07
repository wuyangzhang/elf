#include <iostream>

#include "dataset/VideoLoader.h"
#include "config/Config.h"

#include "test/test.h"
#include "partitioning/PartitionSimulator.h"

int main() {
//    Config config = Config();
//    VideoLoader videoLoader(config);
//    cv::Mat img;
//    while(videoLoader.hasNext()){
//        videoLoader.next(img);
//        std::vector<uchar> buf;
//        imencode(img, buf);
//        std::cout << buf.size() << std::endl;
//        //videoLoader.display(img);
//    }

//    test_network();

//    test_img_transfer();

//    test_parallel_offload();

//    test_multi_sockets();

//    test_elf();

//    PartitionSimulator simulator = PartitionSimulator();
//    simulator.run();

    test_prediction();
    return 0;
}

