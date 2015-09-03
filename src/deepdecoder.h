
#pragma once

#include <boost/shared_ptr.hpp>
#include <caffe/data_layers.hpp>
#include <biotracker/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/common/Grid.h>
#include <boost/logic/tribool.hpp>


namespace deepdecoder {
    const size_t TAG_SIZE = 60;
    const size_t TAG_PIXELS = TAG_SIZE * TAG_SIZE;
    const cv::Point2i TAG_CENTER = cv::Point2i(TAG_SIZE/2, TAG_SIZE/2);

    template<typename Dtype>
    using MemoryDataLayerSPtr = boost::shared_ptr<caffe::MemoryDataLayer<Dtype>>;

    template<typename Dtype>
    using dataset_t = std::pair<std::vector<cv::Mat>, std::vector<std::vector<Dtype>>>;

    template<typename Dtype>
    std::vector<Dtype> triboolIDtoVector(const Grid::idarray_t id) {
        std::vector<Dtype> id_vec;
        for(size_t i = 0; i < id.size(); i++) {
            if(id[i]) {
                id_vec.emplace_back(1);
            } else if(id[i] == boost::tribool::indeterminate_value ){
                id_vec.emplace_back(0.5);
            }else {
                id_vec.emplace_back(0);
            }
        }
        return id_vec;
    }
}
