//
// Created by wuyang zhang on 7/4/20.
//

#ifndef ELF_CONFIG_H
#define ELF_CONFIG_H

#include <iostream>

class Config {
public:

    // computer vision applications
    const std::string apps[3] = {"instanceSegmentation", "objectClassification", "keyPointDetection"};

    const std::string HOME_ADDR = "";

    // application dataset

    // edge servers configurations
    //std::vector<std::string> SERVERS_ADDR{"", "", ""};
    const std::string SERVERS_ADDR[3] = {"127.0.0.1:5050", "127.0.0.1:5051", "127.0.0.1:5052"};
    const std::string LRC_SERVER_ADDR = "";

    const int TOTAL_SERVER_NUM = 3;

    // video datasets
    const std::string DATASETS[3];

    const std::string VIDEO_DIR = "/Users/wuyang/Desktop/photo";

    // video datasets configuration
    int FRAME_HEIGHT = 375;

    int FRAME_WIDTH = 1242;

    const bool USE_LOCAL = true;

    // prediction module
    const std::string MODELS[2] = {"lstm", "attn_lstm"};

    const std::string MODELS_PATH[2] = {"/Users/wuyang/CLionProjects/Elf/prediction/models/predict.pt",
                                        "/Users/wuyang/CLionProjects/Elf/prediction/models/predict.pt"};

    const int MODEL_INDEX = 1;

    const std::string MODEL = MODELS[MODEL_INDEX];

    const std::string MODEL_PATH = MODELS_PATH[MODEL_INDEX];


    /* WINDOW_SIZE
     * Use the last K processing results to predict the RPs in the current video frame.
     */
    const int WINDOW_SIZE = 2;

    const int PADDING_LEN = 160;

    const int BATCH_SIZE = 16;

    // LRC module
    const double LRC_RATIO = 0.5;

    /* LRC_INTERVAL
     * Send LRC request every K frames. A larger K can reduce the workload of a LRC server while a smaller K can
     * identify new appearing objects earlier.
     */
    static const int LRC_INTERVAL = 3;

    // partitioning module
    static constexpr double RP_RESCALE_RATIO = 0.05;

    static const int PAR_NUM = 3;

};


#endif //ELF_CONFIG_H
