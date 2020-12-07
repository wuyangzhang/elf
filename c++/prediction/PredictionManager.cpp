//
// Created by wuyang zhang on 7/5/20.
//

#include "PredictionManager.h"
#include <torch/script.h>
#include <chrono>


PredictionManager::PredictionManager() {
    this->modelPath = config.MODELS_PATH[config.MODEL_INDEX];
    loadModel(this->modelPath);

    // only for the test purpose.
    loadTensors("/Users/wuyang/PycharmProjects/ParSim/raw_txt/", gtTensors);

}

void PredictionManager::predict() {

    preProcess(predictedRP);

    std::cout << predictedRP.sizes() << std::endl;
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(predictedRP);

    auto start = std::chrono::high_resolution_clock::now();
    this->model.forward(inputs);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Prediction runtime: "
              << duration.count() << " microseconds" << std::endl;

    postProcess(predictedRP);
}

/*
 * pre-process the historical RPs and make them ready for the prediction.
 *  1. padding them to the same length.
 *  2. normalize
 *  3. RP indexing
 */
void PredictionManager::preProcess(torch::Tensor &tensor) {

    int maxLen = 0;
    for (auto it = histRPs.begin(); it < histRPs.end(); it++) {
        maxLen = std::max(maxLen, (int) (*it).size(0));
    }

    std::vector<torch::Tensor> paddedTensors;
    for (auto it = histRPs.begin(); it < histRPs.end(); it++) {
        torch::Tensor padding = torch::constant_pad_nd((*it), torch::IntArrayRef{0, 0, 0, maxLen - (*it).size(0)}, 0);
//        std::cout << "before padding shape: " << (*it).sizes() << " after padding shape " << padding.sizes() << std::endl;
        paddedTensors.push_back(padding);
    }

    //at::TensorList tensorList();

    // stack the padding tensors into a single one. For example, it is in the shape of  (WINDOW, 16, 4)
    tensor = torch::stack(paddedTensors).to(torch::kFloat32);

    //tensor.index_put({torch::indexing::Slice(torch::indexing::None, torch::indexing::None, 0)}, )
    for (int i = 0; i < tensor.size(0); i++) {
        for (int j = 0; j < tensor.size(1); j++) {
            tensor[i][j][0] /= 1242;
            tensor[i][j][1] /= 375;
            tensor[i][j][2] /= 1242;
            tensor[i][j][3] /= 375;
        }
    }

    // rp index
    rpIndex(tensor);
    // change the shape from [num, window, 4] => [window, num, 4]
    //    torch::Tensor shape = torch::Tensor([3, 4, 5]);
    //    x.reshape({x.size(0), 784})

    tensor = tensor.reshape({tensor.size(1), tensor.size(0), tensor.size(2)});
    // to fix in the future. currently assume the batch size == 1;
    //tensor = tensor.reshape( {tensor.size(2), tensor.size(1), tensor.size(3)});

}

/*
 * Refers to Section 4.2 RP indexing in the paper.
 * This function will best match the historical objects with each object in the last frame.
 */
void PredictionManager::rpIndex(torch::Tensor &input) {

    // reshape the input if it the inference input only with three dimensions.
//    if (tensor.sizes().size() == 3) {
//        tensor = tensor.reshape({1, tensor.size(0), tensor.size(1), tensor.size(2)});
//    }

    // tensor shape: batch size, window size, count, 4
    // int batchSize = tensor.size(0);
    int WINDOW = input.size(0);
    int RPCount = input.size(1);

    // calculate the areas
    torch::Tensor areas = torch::zeros({WINDOW, RPCount});
    for (int w = 0; w < WINDOW; w++) {
        for (int r = 0; r < RPCount; r++) {
            areas[w][r] = (input[w][r][2] - input[w][r][0]) * (input[w][r][3] - input[w][r][1]);
        }
    }

    torch::Tensor xOffset = torch::zeros({RPCount, WINDOW, RPCount});

    torch::Tensor yOffset = torch::zeros({RPCount, WINDOW, RPCount});

    torch::Tensor areaOffset = torch::zeros({RPCount, WINDOW, RPCount});

    torch::Tensor metrics = torch::zeros({RPCount, WINDOW, RPCount});

    for (int r = 0; r < RPCount; r++) {
        for (int w = 0; w < WINDOW; w++) {
            for (int k = 0; k < RPCount; k++) {
                xOffset[r][w][k] = at::abs((input[w][r][0] + input[w][r][2]) - (input[w][k][0] + input[w][k][2])) / 2;
                yOffset[r][w][k] = at::abs((input[w][r][3] + input[w][r][1]) - (input[w][k][3] + input[w][k][1])) / 2;
                areaOffset[r][w][k] = at::abs(at::sqrt(areas[w][r]) - at::sqrt(areas[w][k]));
                metrics[r][w][k] = xOffset[r][w][k] + yOffset[r][w][k] + areaOffset[r][w][k];
            }
        }
    }

    torch::Tensor index = torch::zeros({WINDOW, RPCount});
//    torch::Tensor minVals = torch::zeros({WINDOW, RPCount});

    for (int r = 0; r < RPCount; r++) {
        for (int w = 0; w < WINDOW; w++) {
            float minVal = 10000000;
            int minIndex = 0;
            for (int k = 0; k < RPCount; k++) {
                float val = metrics[r][w][k].item().toFloat();
                if (val < minVal) {
                    minVal = val;
                    minIndex = k;
                }
            }
            index[w][r] = minIndex;
        }

    }

    torch::Tensor output = torch::zeros(input.sizes());
    float THRESHOLD = 0.02;

    for (int w = 0; w < WINDOW; w++) {
        for (int k = 0; k < RPCount; k++) {
            int matchIndex = index[w][k].item().toInt();
            if (metrics[k][w][matchIndex].item().toFloat() < THRESHOLD) {
                output[w][k] = input[w][matchIndex];
            }
        }
    }

    // post-processing.
    // corner case 1: if find any empty RPs in the last frame, we set the RP in prev frames to empty also.
    torch::Tensor zeroMask = torch::zeros(RPCount);
    for (int r = 0; r < RPCount; r++) {
        if (input[-1][r][0].item().toFloat() + input[-1][r][1].item().toFloat() + input[-1][r][2].item().toFloat() +
            input[-1][r][3].item().toFloat() == 0) {
            zeroMask[r] = 1;
        }
    }


    for (int r = 0; r < RPCount; r++) {
        bool flag = zeroMask[r].item().toInt() == 1;
        for (int w = 0; w < WINDOW; w++) {
            if (flag) {
                output[w][r][0] = output[w][r][1] = output[w][r][2] = output[w][r][3] = 0;
            }
        }

    }

    // corner case 2: if locate an non-zero RP in the last frame while the prev frames has a zero RP in the corresponding position,
    // we set the previous frames to the same RP value
    for (int r = 0; r < RPCount; r++) {
        for (int w = 0; w < WINDOW; w++) {
            if (output[w][r][0].item().toFloat() + output[w][r][1].item().toFloat() + output[w][r][2].item().toFloat() +
                output[w][r][3].item().toFloat() == 0 and
                output[-1][r][0].item().toFloat() + output[-1][r][1].item().toFloat() +
                output[-1][r][2].item().toFloat() + output[-1][r][3].item().toFloat() != 0) {
                output[w][r] = output[-1][r];
            }
        }
    }

    input = output;
}

void PredictionManager::extendRP(torch::Tensor &tensor) {
    for (int i = 0; i < tensor.size(0); i++) {
        double xDiff = tensor[i][2].item().toDouble() - tensor[i][0].item().toDouble();
        double yDiff = tensor[i][3].item().toDouble() - tensor[i][1].item().toDouble();

        tensor[i][0] = std::max(0., tensor[i][0].item().toDouble() - xDiff * RP_RESCALE_RATIO);
        tensor[i][1] = std::max(0., tensor[i][1].item().toDouble() - yDiff * RP_RESCALE_RATIO);
        tensor[i][2] = std::min((double) config.FRAME_WIDTH, tensor[i][2].item().toDouble() + xDiff * RP_RESCALE_RATIO);
        tensor[i][3] = std::max((double) config.FRAME_HEIGHT,
                                tensor[i][3].item().toDouble() + xDiff * RP_RESCALE_RATIO);
    }
}


void PredictionManager::postProcess(torch::Tensor &tensor) {
    for (int i = 0; i < tensor.size(0); i++) {
        tensor[i][0] *= 1242;
        tensor[i][1] *= 375;
        tensor[i][2] *= 1242;
        tensor[i][3] *= 375;
    }
    extendRP(tensor);
}


bool PredictionManager::isActive() {
    return (int) this->histRPs.size() >= config.WINDOW_SIZE;
}

void PredictionManager::loadModel(std::string modelPath) {

    std::shared_ptr<torch::jit::script::Module> module;

    try {
        this->model = torch::jit::load(modelPath);
    } catch (const c10::Error &e) {
        std::cerr << "error loading the model from " << modelPath << std::endl;
    }
}

void PredictionManager::moveModelToCuda(std::string deviceId) {
    // device id: "cuda:0"
    this->model.to(at::Device(deviceId));
}

void PredictionManager::addHistTensors(torch::Tensor tensor) {
    // skip if no object has been detected.
    if (tensor.dim() == 0) {
        return;
    }
    if ((int) this->histRPs.size() == this->MAX_QUEUE_SIZE) {
        this->histRPs.pop_front();
    }

    this->histRPs.push_back(tensor);
}

torch::Tensor PredictionManager::getPredictedRP() {
    return this->predictedRP;
}

torch::Tensor PredictionManager::getGTRP(int index) {
    return gtTensors.at(index);
}

void PredictionManager::printModelInfo() {
    //this->model.
    //std::cout << this->model << std::endl;
}

