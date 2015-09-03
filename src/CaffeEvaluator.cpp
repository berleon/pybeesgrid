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


}
