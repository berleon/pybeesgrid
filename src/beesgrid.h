
#pragma once

#include <boost/shared_ptr.hpp>
#include "Grid.h"
#include <boost/logic/tribool.hpp>


namespace beesgrid {
    const size_t TAG_SIZE = 64;
    const size_t TAG_PIXELS = TAG_SIZE * TAG_SIZE;
    const cv::Point2i TAG_CENTER = cv::Point2i(TAG_SIZE/2, TAG_SIZE/2);
    const size_t NUM_GRID_CONFIGS = 6;
    const size_t NUM_MIDDLE_CELLS = Grid::NUM_MIDDLE_CELLS;

    template<typename Dtype>
    std::vector<Dtype> triboolIDtoVector(const Grid::idarray_t & id) {
        std::vector<Dtype> id_vec;
        for(size_t i = 0; i < id.size(); i++) {
            if(id[i]) {
                id_vec.emplace_back(1);
            } else if(! id[i]){
                id_vec.emplace_back(0);
            } else {
                id_vec.emplace_back(0.5);
            }
        }
        return id_vec;
    }

    std::string getLabelsAsString(const Grid::idarray_t & id_arr);
}
