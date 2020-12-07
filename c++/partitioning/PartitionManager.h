//
// Created by wuyang zhang on 7/5/20.
//

#ifndef ELF_PARTITIONMANAGER_H
#define ELF_PARTITIONMANAGER_H

#include <iostream>
#include <vector>

#include "../utils/utils.h"
#include "../config/Config.h"

class PartitionManager {
public:
    PartitionManager();

    //~PartitionManager();

    /* Partition a frame based on the coordinates of RP boxes.
     * Need to run findRPBoxes() or getEqualRPBoxes() to update the RP boxes.
     * The partitioning results will be placed in this->framePartitions.
     *
     */
    void partitionFrame(const cv::Mat &frame);

    void findRPBoxes(const cv::Mat &frame, const std::vector<float> &proc_latency, std::vector<RP> &rps);

    void getEqualRPBoxes(cv::Mat &frame);

    std::vector<RP> *getRPBoxCoords();

    std::vector<cv::Mat> *getFramePartitions();


private:

    static void findRPsBoundary(const std::vector<RP> &rps, int &, int &, int &, int &);

    static void findRPsBoundary(const std::vector<RP*> &rps, int &, int &, int &, int &);

    void initBoxes(std::vector<RP> &rps, int&, int&) const;

    static void matchRPBox(std::vector<RP> &rps, std::vector<RP> &rpBoxes, std::map<RP*, std::vector<RP*>> &rpMatch);

    void adjustRPBox(std::vector<RP> &rpBoxes, std::map<RP*, std::vector<RP*>> &index);

    void rescaleRPBox(std::vector<RP> &rpBoxes) const;

    struct Offset {
        int x;
        int y;
        Offset(){
            x = 0;
            y = 0;
        }
        Offset(int a, int b): x(a), y(b){}
    };

    void findPartitionOffset();

    std::vector<Offset> parOffsets;

    std::vector<RP> rpBoxCoords;

    std::vector<cv::Mat> framePartitions;

    const float rescaleRatio = Config::RP_RESCALE_RATIO;
    const int parNum = Config::PAR_NUM;

    int frameWidth{}, frameHeight{};

    int parUnit = 0;
    int parHeight = 0;

};


#endif //ELF_PARTITIONMANAGER_H
