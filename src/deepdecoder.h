
#pragma once

#include <boost/shared_ptr.hpp>
#include <caffe/data_layers.hpp>

namespace deepdecoder {
    const size_t TAG_SIZE = 60;

    template<typename Dtype>
    using MemoryDataLayerSPtr = boost::shared_ptr<caffe::MemoryDataLayer<Dtype>>;

    using dataset_t = std::pair<std::vector<cv::Mat>, std::vector<int>>;
}
