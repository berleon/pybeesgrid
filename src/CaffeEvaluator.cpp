#include "CaffeEvaluator.h"

#include <opencv2/highgui/highgui.hpp>
#include <cereal/archives/json.hpp>
#include <caffe/util/io.hpp>
#include <biotracker/tracking/serialization/SerializationData.h>
#include <biotracker/tracking/algorithm/BeesBook/ImgAnalysisTracker/legacy/Grid3D.h>
#include <caffe/data_layers.hpp>
#include <QtGui/qvector3d.h>

#include "deepdecoder.h"

namespace deepdecoder {
CaffeEvaluator::CaffeEvaluator(std::vector<std::string> && gt_files) :
    _gt_files(std::move(gt_files))
{
}

cv::Mat CaffeEvaluator::loadGTImage(const std::string &gt_path) {
    return cv::imread(gt_path + ".jpeg", CV_LOAD_IMAGE_GRAYSCALE);
}

std::vector<GroundTruthGridSPtr> CaffeEvaluator::loadGTData(
        const std::string &gt_path) {

    std::ifstream is(gt_path + ".tdat");

    cereal::JSONInputArchive ar(is);

    Serialization::Data data;
    ar(data);
    return getPipelineGridsForFrame(data, 0);
}

dataset_t CaffeEvaluator::toDataset(
        const std::vector<GroundTruthGridSPtr> &gt_grids,
        const cv::Mat &image) {

    std::vector<cv::Mat> images;
    std::vector<int> labels;
    for(const auto & gt_grid : gt_grids) {
        const auto center = gt_grid->getCenter();
        cv::Rect box{center.x - int(TAG_SIZE / 2), center.y - int(TAG_SIZE / 2),
                     TAG_SIZE, TAG_SIZE};
        if(box.x > 0 && box.y > 0 && box.x + box.width < image.cols && box.y + box.height < image.rows) {
            // TODO:: add border
            images.emplace_back(image(box).clone());
            labels.push_back(gt_grid->getId());
        }
    }
    return std::make_pair<std::vector<cv::Mat>, std::vector<int>>(std::move(images), std::move(labels));
}

dataset_t CaffeEvaluator::getAllData() {
    dataset_t total_dataset;
    auto &mats = total_dataset.first;
    auto & labels = total_dataset.second;
    for(const auto & gt_file : _gt_files) {
        const auto gt_grids = loadGTData(gt_file);
        const auto image = loadGTImage(gt_file);
        auto dataset = toDataset(gt_grids, image);
        ;
        mats.insert(mats.cbegin(), std::make_move_iterator(dataset.first.begin()),
                 std::make_move_iterator(dataset.first.end()));

        labels.insert(labels.cbegin(), std::make_move_iterator(dataset.second.begin()),
                    std::make_move_iterator(dataset.second.end()));
    }
    return total_dataset;
}
}
