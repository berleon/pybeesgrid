#pragma once

#include <string>
#include <caffe/net.hpp>
#include <biotracker/tracking/algorithm/BeesBook/ImgAnalysisTracker/GroundTruthEvaluator.h>
#include "deepdecoder.h"
#include "CaffeGridGenerator.h"

namespace deepdecoder {

struct CaffeEvaluationResult {
    double accuracy;
};

class CaffeEvaluator {
    const std::vector<std::string> _gt_files;
public:
    CaffeEvaluator(std::vector<std::string> && gt_files);
    template<typename Dtype>
    CaffeEvaluationResult evaluate(caffe::Net<Dtype> & net) {
        size_t n_total = 0;
        size_t n_right = 0;
        size_t n_false = 0;
        for(const auto & gt_file : _gt_files) {
            const auto gt_grids = loadGTData(gt_file);
            const auto image = loadGTImage(gt_file);
            const auto dataset = toDataset(gt_grids, image);
            auto memory_layer = extractMemoryDataLayer(&net);
            memory_layer->set_batch_size(dataset.first.size());
            net.Reshape();
            memory_layer->AddMatVector(dataset.first, dataset.second);
            const auto pred_labels = topLabelPrefilled(net);
            for(size_t i = 0; i < dataset.second.size(); i++) {
                n_total++;
                if (pred_labels.at(i) == dataset.second.at(i)) {
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

    dataset_t getAllData();
private:
    std::vector<GroundTruthGridSPtr> loadGTData(const std::string & gt_path);
    cv::Mat loadGTImage(const std::string & gt_path);
    dataset_t toDataset(const std::vector<GroundTruthGridSPtr> &gt_data,
                        const cv::Mat &img);

    template<typename Dtype>
    std::vector<int> topLabelPrefilled(caffe::Net<Dtype> & net) {
        const auto pred_labels = forward(net);
        std::vector<int> int_pred_labels;
        for(size_t i = 0; i < pred_labels.size(); i++) {
            const auto pred_probs = pred_labels.at(i);
            size_t max_index = 0;
            double max_prob = 0;
            for(size_t j = 0; j < pred_labels.size(); j++) {
                if(pred_probs.at(j) > max_prob) {
                    max_index = j;
                    max_prob = pred_probs.at(j);
                }
            }
            int_pred_labels.push_back(max_index);
        }
        return int_pred_labels;
    }

    template<typename Dtype>
    std::vector<std::vector<Dtype>> forward(caffe::Net<Dtype> &net) {
        net.ForwardPrefilled();
        CHECK(net.output_blobs().size() == 1);
        caffe::Blob<Dtype>*output_blob = net.output_blobs()[0];
        size_t n_labels = output_blob->shape(1);
        std::vector<std::vector<Dtype>> outputs;
        outputs.reserve(n_labels);
        for(int i = 0; i < output_blob->shape(0); i++) {
            const Dtype* begin = output_blob->cpu_data() + n_labels*i;
            const Dtype* end = begin + n_labels;
            outputs.emplace_back(std::vector<Dtype>(begin, end));
        }
        return outputs;
    }
};


}
