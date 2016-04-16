
#pragma once

#include <pipeline/common/Grid.h>

#include "beesgrid.h"

namespace beesgrid {
class GridGenerator;

class GeneratedGrid : public Grid {
public:
    explicit GeneratedGrid(Grid::idarray_t id, cv::Point2i center, double radius,
                           double angle_x, double angle_y, double angle_z);
    explicit GeneratedGrid(Grid::idarray_t id, cv::Point2i center, double radius,
                           double angle_x, double angle_y, double angle_z,
                           const std::shared_ptr<Grid::Structure> structure);
    explicit GeneratedGrid(Grid::idarray_t id, cv::Point2i center, double radius,
                           double angle_x, double angle_y, double angle_z,
                           const std::shared_ptr<Grid::Structure> structure,
                           const std::shared_ptr<Grid::coordinates3D_t> coordinates3D);

    virtual ~GeneratedGrid() override = default;
    int getLabelAsInt() const;

    template<typename Dtype>
    inline std::vector<Dtype> getLabelAsVector() const {
        return triboolIDtoVector<Dtype>(_ID);
    }
    inline std::string getLabelsAsString() const {
        return beesgrid::getLabelsAsString(_ID);
    }
    inline const std::vector<std::vector<cv::Point>> & getCoordinates2D() const {
        return _coordinates2D;
    }
    GeneratedGrid scale(double factor) const;

    inline const std::shared_ptr<Grid::Structure> structure() const {
        return _structure;
    }
};

}
