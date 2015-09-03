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
        for(const auto & gt_file : this->_gt_files) {
            const auto gt_grids = this->loadGTData(gt_file);
            const auto image = this->loadGTImage(gt_file);
            const auto dataset = this->toDataset(gt_grids, image);
            auto memory_layer = extractMemoryDataLayer(&net);
            memory_layer->set_batch_size(dataset.first.size());
            net.Reshape();
            memory_layer->AddMatVector(dataset.first, flatten(dataset.second));
            const auto pred_labels = bitsPrefilled(net);
            for(size_t i = 0; i < dataset.second.size(); i++) {
                n_total++;
                for(const auto & l: pred_labels.at(i)) {

                    std::cout << l << ",";
                }
                std::cout << std::endl;
                if (isPredictedRight<Dtype>(pred_labels.at(i), dataset.second.at(i))) {
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
        for(const auto & gt_file : this->_gt_files) {
            const auto gt_grids = this->loadGTData(gt_file);
            const auto image = this->loadGTImage(gt_file);
            auto dataset = this->toDataset(gt_grids, image);
            mats.insert(mats.cbegin(), std::make_move_iterator(dataset.first.begin()),
                        std::make_move_iterator(dataset.first.end()));

            labels.insert(labels.cbegin(), std::make_move_iterator(dataset.second.begin()),
                          std::make_move_iterator(dataset.second.end()));
        }
        return total_dataset;
    }
private:
    dataset_t<Dtype> toDataset(const std::vector<GroundTruthGridSPtr> &gt_grids,
              const cv::Mat &image) {
        std::vector<cv::Mat> images;
        std::vector<std::vector<Dtype>> labels;
        for(const auto & gt_grid : gt_grids) {
            const auto center = gt_grid->getCenter();
            cv::Rect box{center.x - int(TAG_SIZE / 2), center.y - int(TAG_SIZE / 2),
                         TAG_SIZE, TAG_SIZE};
            if(box.x > 0 && box.y > 0 && box.x + box.width < image.cols && box.y + box.height < image.rows) {
                // TODO:: add border
                images.emplace_back(image(box).clone());
                std::vector<float> bits;
                for(const auto & bit : triboolIDtoVector<Dtype>(gt_grid->getIdArray())) {
                    bits.push_back(bit);
                }
                labels.push_back(bits);
            }
        }
        return std::make_pair<
                std::vector<cv::Mat>,
                std::vector<std::vector<Dtype>>
        >(std::move(images), std::move(labels));
    }
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
