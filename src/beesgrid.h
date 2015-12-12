
#pragma once

#include <boost/shared_ptr.hpp>
#include <pipeline/common/Grid.h>
#include <boost/logic/tribool.hpp>
#include <pipeline/util/GroundTruthEvaluator.h>
#include "Grid3D.h"

namespace beesgrid {
    const size_t TAG_SIZE = 64;
    const size_t TAG_PIXELS = TAG_SIZE * TAG_SIZE;
    const cv::Point2i TAG_CENTER = cv::Point2i(TAG_SIZE/2, TAG_SIZE/2);
    const size_t NUM_GRID_CONFIGS = 6;

    template<typename Dtype>
    std::vector<Dtype> triboolIDtoVector(const Grid::idarray_t & id) {
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

    std::string getLabelsAsString(const Grid::idarray_t & id_arr);


    struct GroundTruthDatum {
        std::vector<float> bits;
        float z_rot;
        float y_rot;
        float x_rot;
        float x;
        float y;
        float radius;

        inline static GroundTruthDatum fromGrid3D(const GroundTruthGridSPtr & grid) {
            std::vector<float> bits = triboolIDtoVector<float>(grid->getIdArray());
            return GroundTruthDatum{
                bits,
                static_cast<float>(grid->getZRotation()),
                static_cast<float>(grid->getYRotation()),
                static_cast<float>(grid->getXRotation()),
                static_cast<float>(grid->getCenter().x),
                static_cast<float>(grid->getCenter().y),
                static_cast<float>(grid->getRadius())
            };
        }
    };

    using gt_dataset_t = std::pair<std::vector<cv::Mat>, std::vector<GroundTruthDatum>>;
}
