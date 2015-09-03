
#pragma once

#include <memory>
#include <caffe/solver.hpp>
#include <caffe/data_layers.hpp>
#include <boost/smart_ptr.hpp>

#include "deepdecoder.h"
#include "GeneratedGrid.h"

namespace deepdecoder {


template<typename Dtype>
void generateGridDataset(size_t batch_size,
                         GridGenerator & gen,
                         std::vector<cv::Mat> * mats,
                         std::vector<Dtype> * labels) {
    for(size_t i = 0; i < batch_size; i++) {
        GeneratedGrid gg = gen.randomGrid();
        mats->emplace_back(gg.cvMat());
        for(const auto & label: gg.getLabelAsVector<Dtype>()) {
            labels->push_back(label);
        }
    }
}

template<typename Dtype>
class CaffeGridGenerator : public caffe::MemoryDataLayer<Dtype>::MatGenerator {
public:
    explicit CaffeGridGenerator() { }
    virtual ~CaffeGridGenerator() = default;
    virtual void generate(int batch_size, std::vector<cv::Mat> * mats, std::vector<Dtype> * labels) {
        return generateGridDataset(batch_size, _gen, mats, labels);
    }
private:
    GridGenerator _gen;
};

template<typename Dtype>
inline MemoryDataLayerSPtr<Dtype> extractMemoryDataLayer(caffe::Net<Dtype> * net) {
    auto top_layer = net->layers()[0];
    CHECK(std::string("MemoryData").compare(top_layer->type()) == 0);
    return boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype>>(top_layer);
}

template<typename Dtype>
void addGridGeneratorToMemoryDataLayer(caffe::Solver<Dtype> &solver) {
    auto grid_gen = boost::make_shared<CaffeGridGenerator<Dtype>>();
    extractMemoryDataLayer<Dtype>(solver.net().get())->SetMatGenerator(grid_gen);
    for(auto & net : solver.test_nets()) {
        extractMemoryDataLayer<Dtype>(net.get())->SetMatGenerator(grid_gen);
    }
}

template<typename Dtype>
void addGridGeneratorToMemoryDataLayer(caffe::Net<Dtype> &net) {
    auto grid_gen = boost::make_shared<CaffeGridGenerator<Dtype>>();
    extractMemoryDataLayer<Dtype>(&net)->SetMatGenerator(grid_gen);
}

}
