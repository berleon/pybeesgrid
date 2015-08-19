
#pragma once

#include <memory>
#include <caffe/solver.hpp>
#include <caffe/data_layers.hpp>
#include <boost/smart_ptr.hpp>

#include "deepdecoder.h"

namespace deepdecoder {

class GeneratedGrid;

void generateGridDataset(size_t batch_size, std::vector<cv::Mat> * mats, std::vector<int> * labels);


template<typename Dtype>
class GridGenerator : public caffe::MemoryDataLayer<Dtype>::MatGenerator {
public:
    explicit GridGenerator() {}
    virtual ~GridGenerator() = default;
    virtual void generate(int batch_size, std::vector<cv::Mat> * mats, std::vector<int> * labels) {
        return generateGridDataset(batch_size, mats, labels);
    }
};

template<typename Dtype>
inline MemoryDataLayerSPtr<Dtype> extractMemoryDataLayer(caffe::Net<Dtype> * net) {
    auto top_layer = net->layers()[0];
    CHECK(std::string("MemoryData").compare(top_layer->type()) == 0);
    return boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype>>(top_layer);
}

template<typename Dtype>
void addGridGeneratorToMemoryDataLayer(caffe::Solver<Dtype> &solver) {
    auto grid_gen = boost::make_shared<GridGenerator<Dtype>>();
    extractMemoryDataLayer<Dtype>(solver.net().get())->SetMatGenerator(grid_gen);
    for(auto & net : solver.test_nets()) {
        extractMemoryDataLayer<Dtype>(net.get())->SetMatGenerator(grid_gen);
    }
}

}
