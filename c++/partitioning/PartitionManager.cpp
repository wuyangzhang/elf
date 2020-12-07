//
// Created by wuyang zhang on 7/5/20.
//

#include "PartitionManager.h"
#include <map>

PartitionManager::PartitionManager() {
    for (int i = 0; i < this->parNum; i++) {
        rpBoxCoords.emplace_back();
    }
};


/* Partition a frame based on the coordinates of RP boxes.
 * Need to run findRPBoxes() or getEqualRPBoxes() to update the RP boxes.
 * The partitioning results will be placed in this->framePartitions.
 *
 */
void PartitionManager::partitionFrame(const cv::Mat &frame) {
    this->framePartitions.clear();
    for (auto & rpBoxCoord : this->rpBoxCoords) {
        this->framePartitions.emplace_back(
                frame(cv::Rect(rpBoxCoord.x1, rpBoxCoord.y1, rpBoxCoord.x2 - rpBoxCoord.x1, rpBoxCoord.y2 - rpBoxCoord.y1)));
    }
}


void PartitionManager::findRPBoxes(const cv::Mat &frame, const std::vector<float> &proc_latency, std::vector<RP> &rps) {
    /*A frame partition scheme.
    Multi-Capacity Bin Packing problem.
            This frame partition scheme performs based on the position of bounding boxes(bbox),
            the weights of bbox that indicate the potential computing costs, and
    the available computing resources that are represented by the historical
    computing time.
            step 1. Equal partition.
            step 2. Computation complexity aware placement.
    for each bbox, check whether it is overlapped with multiple partitions.
    if not, add it to that partition and change the partition weight.
    if yes, select one of partitions based on its current weight. Each partition should
    have equal probability to be selected.
    :param frame: the target frame to be partition.
    :param rps: all the region proposals along with their coordinates!
    :param cost_weights:
    :param proc_latency:
    :return N partitions
    */

    this->frameWidth = frame.cols;
    this->frameHeight = frame.rows;

    // initialize par boxes by equally partitioning the frame.
    // the number of partitions equals to the number of available servers.

    // find the boundary of all the rps.
    int min_x1, min_y1, max_x2, max_y2;

    findRPsBoundary(rps, min_x1, min_y1, max_x2, max_y2);

    this->parUnit = (int) (max_x2 - min_x1) / this->parNum;
    this->parHeight = (int) max_y2 - min_y1;
    // initialize RP boxes
    this->initBoxes(rpBoxCoords, min_x1, min_y1);

    // find associated RP boxes for each RP
    std::map<RP *, std::vector<RP *>> rpMatch;
    PartitionManager::matchRPBox(rps, rpBoxCoords, rpMatch);

    // rescale each partitioning box in order to fully cover its associated RPs.
    this->adjustRPBox(rpBoxCoords, rpMatch);

    this->rescaleRPBox(rpBoxCoords);
}


void PartitionManager::getEqualRPBoxes(cv::Mat &frame) {
    this->frameWidth = frame.cols;
    this->frameHeight = this->parHeight = frame.rows;

    this->parUnit = (int) this->frameWidth / this->parNum;

    int min_x1 = 0, min_y1 = 0;
    this->initBoxes(rpBoxCoords, min_x1, min_y1);
}

void PartitionManager::findRPsBoundary(const std::vector<RP> &rps, int &min_x1, int &min_y1, int &max_x2,
                                       int &max_y2) {
    min_x1 = min_y1 = 10000;
    max_x2 = max_y2 = 0;

    for (const auto & rp : rps) {
        if (min_x1 > rp.x1) {
            min_x1 = rp.x1;
        }

        if (min_y1 > rp.y1) {
            min_y1 = rp.y1;
        }

        if (max_x2 < rp.x2) {
            max_x2 = rp.x2;
        }

        if (max_y2 < rp.y2) {
            max_y2 = rp.y2;
        }
    }
}

void PartitionManager::findRPsBoundary(const std::vector<RP *> &rps, int &min_x1, int &min_y1, int &max_x2,
                                       int &max_y2) {
    min_x1 = min_y1 = 10000;
    max_x2 = max_y2 = 0;

    for (auto rp : rps) {
        if (min_x1 > rp->x1) {
            min_x1 = rp->x1;
        }

        if (min_y1 > rp->y1) {
            min_y1 = rp->y1;
        }

        if (max_x2 < rp->x2) {
            max_x2 = rp->x2;
        }

        if (max_y2 < rp->y2) {
            max_y2 = rp->y2;
        }
    }
}


void PartitionManager::initBoxes(std::vector<RP> &rps, int &min_rp_x1, int &min_rp_y1) const {
    for (int i = 0; i < this->parNum; i++) {
        rps.at(i).x1 = i * this->parUnit + min_rp_x1;
        rps.at(i).y1 = min_rp_y1;
        rps.at(i).x2 = i * this->parUnit + this->parUnit + min_rp_x1;
        rps.at(i).y2 = min_rp_y1 + this->parHeight;
    }
}


void
PartitionManager::matchRPBox(std::vector<RP> &rps, std::vector<RP> &rpBoxes,
                             std::map<RP *, std::vector<RP *>> &rpMatch) {
    /**
     * find the matching RP box for each RP.
     * @param rps
     * @param rpBoxes
     * @param rpMatch: map, key: rpBox, value: associated rps
    */
    for (auto itr = rps.begin(); itr < rps.end(); itr++) {
        RP *matchBox;
        int overlap = 0;
        // find the maximal overlap
        for (auto itb = rpBoxes.begin(); itb < rpBoxes.end(); itb++) {
            int x1 = std::max((*itr).x1, (*itb).x1);
            int y1 = std::max((*itr).y1, (*itb).y1);
            int x2 = std::min((*itr).x2, (*itb).x2);
            int y2 = std::min((*itr).y2, (*itb).y2);
            int area;
            int dx = x2 - x1;
            int dy = y2 - y1;
            if (dx > 0 and dy > 0) {
                area = dx * dy;
                if (area > overlap) {
                    overlap = area;
                    matchBox = &(*itb);
                }
            }
        }
        rpMatch[matchBox].push_back(&(*itr));
    }
}

void PartitionManager::adjustRPBox(std::vector<RP> &rpBoxes, std::map<RP *, std::vector<RP *>> &index) {
    for (auto it = rpBoxes.begin(); it < rpBoxes.end(); it++) {
        if (index[&(*it)].empty()) {
            (*it).x1 = (*it).y1 = 0;
            (*it).x2 = (*it).y2 = 5;
        } else {
            this->findRPsBoundary(index[&(*it)], (*it).x1, (*it).y1, (*it).x2, (*it).y2);
        }
    }
}

void PartitionManager::rescaleRPBox(std::vector<RP> &rpBoxes) const {
    for (auto it = rpBoxes.begin(); it < rpBoxes.end(); it++) {
        int dx = ((*it).x2 - (*it).x1) * this->rescaleRatio;
        int dy = ((*it).y2 - (*it).y1) * this->rescaleRatio;

        (*it).x1 = std::max((*it).x1 - dx, 0);
        (*it).y1 = std::max((*it).y1 - dy, 0);
        (*it).x2 = std::min((*it).x2 + dx, this->frameWidth);
        (*it).y2 = std::min((*it).y2 + dy, this->frameHeight);
    }
}

void PartitionManager::findPartitionOffset() {
    for (auto it = this->rpBoxCoords.begin(); it < this->rpBoxCoords.end(); it++) {
        this->parOffsets.emplace_back((*it).x1, (*it).y1);
    }
}


std::vector<RP> *PartitionManager::getRPBoxCoords() {
    return &this->rpBoxCoords;
}

std::vector<cv::Mat> *PartitionManager::getFramePartitions() {
    return &this->framePartitions;
}
