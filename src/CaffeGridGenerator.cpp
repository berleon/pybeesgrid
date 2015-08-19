
#include "CaffeGridGenerator.h"
#include "GeneratedGrid.h"

namespace deepdecoder {
void generateGridDataset(size_t batch_size, std::vector<cv::Mat> * mats, std::vector<int> * labels) {
    for(size_t i = 0; i < batch_size; i++) {
        GeneratedGrid gg;
        mats->emplace_back(gg.cvMat());
        labels->emplace_back(gg.getLabel());
    }
}
}
