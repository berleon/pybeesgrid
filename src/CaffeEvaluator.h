#pragma once

#include <string>
#include <caffe/net.hpp>
#include <biotracker/tracking/algorithm/BeesBook/ImgAnalysisTracker/GroundTruthEvaluator.h>
#include <biotracker/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/datastructure/PipelineGrid.h>
#include "deepdecoder.h"
#include "CaffeGridGenerator.h"
#include "GroundTruthDataLoader.h"

namespace deepdecoder {

struct CaffeEvaluationResult {
    double accuracy;
};
template<typename Dtype>
bool isPredictedRight(const std::vector<bool> predicted,
                      const std::vector<Dtype> ground_truth) {
    CHECK_EQ(predicted.size(), ground_truth.size());
    for(size_t i = 0; i < predicted.size(); i++) {
        if((predicted.at(i) && ground_truth.at(i) != 1.) ||
           (! predicted.at(i) && ground_truth.at(i) == 1.)) {
            return false;
        }
    }
    return true;
}

template<typename Dtype>
std::vector<Dtype> flatten(std::vector<std::vector<Dtype>> vec) {
    std::vector<Dtype> flatten;
    flatten.reserve(vec.size()*vec.at(0).size());
    for(const auto & subvec: vec) {
        flatten.insert(flatten.cend(), subvec.cbegin(), subvec.cend());
    }
    return flatten;
}

template<typename Dtype>
class CaffeEvaluator : public GroundTruthDataLoader<Dtype> {
public:
    inline explicit CaffeEvaluator(std::vector<std::string> & gt_files) :
            GroundTruthDataLoader<Dtype>(gt_files) { }

    CaffeEvaluationResult evaluate(caffe::Net<Dtype> & net) {
        size_t n_total = 0;
        size_t n_right = 0;
        size_t n_false = 0;
        const size_t batch_size = 64;
        auto memory_layer = extractMemoryDataLayer(&net);
        memory_layer->set_batch_size(batch_size);
        net.Reshape();
        while(auto opt_batch = this->batch(batch_size, GTRepeat::NO_REPEAT)) {
            const auto & batch = opt_batch.get();
            const auto & grids = batch.first;
            const auto & labels = batch.second;
            memory_layer->AddMatVector(grids, flatten(labels));
            const auto pred_labels = bitsPrefilled(net);
            for(size_t i = 0; i < labels.size(); i++) {
                n_total++;
                for(const auto & l: pred_labels.at(i)) {
                    std::cout << l << ",";
                }
                std::cout << std::endl;
                if (isPredictedRight<Dtype>(pred_labels.at(i), labels.at(i))) {
                    n_right++;
                } else {
                    n_false++;
                }
            }
        }
        return CaffeEvaluationResult{
            double(n_right) / double(n_total)
        };
    }

    dataset_t<Dtype> getAllData() {
        dataset_t<Dtype> total_dataset;
        auto &mats = total_dataset.first;
        auto & labels = total_dataset.second;
        while(auto opt_dataset = this->batch(128, NO_REPEAT)) {
            auto dataset = opt_dataset.get();
            const auto images = dataset;
            mats.insert(mats.cbegin(), std::make_move_iterator(dataset.first.begin()),
                        std::make_move_iterator(dataset.first.end()));

            labels.insert(labels.cbegin(), std::make_move_iterator(dataset.second.begin()),
                          std::make_move_iterator(dataset.second.end()));
        }
        return total_dataset;
    }
private:
    std::vector<std::vector<bool>> bitsPrefilled(caffe::Net<Dtype> &net) {
        const auto outputs = forward(net);
        std::vector<std::vector<bool>> bits_predicted;
        for(size_t i = 0; i < outputs.size(); i++) {
            const auto bit_probs = outputs.at(i);
            std::vector<bool> bits;
            for(size_t j = 0; j < bit_probs.size(); j++) {
                bits.push_back(bit_probs.at(j) >= 0.5);
            }
            bits_predicted.push_back(bits);
        }
        return bits_predicted;
    }

    std::vector<std::vector<Dtype>> forward(caffe::Net<Dtype> &net) {
        net.ForwardPrefilled();
        CHECK(net.output_blobs().size() == 1);
        caffe::Blob<Dtype>*output_blob = net.output_blobs()[0];
        size_t bs = output_blob->shape(0);
        size_t n_labels = output_blob->shape(1);
        std::vector<std::vector<Dtype>> outputs;
        for(size_t i = 0; i < bs; i++) {
            const Dtype* begin = output_blob->cpu_data() + n_labels*i;
            const Dtype* end = begin + n_labels;
            outputs.push_back(std::vector<Dtype>(begin, end));
        }
        std::cout << "collected outputs" << std::endl;
        return outputs;
    }
};


}
