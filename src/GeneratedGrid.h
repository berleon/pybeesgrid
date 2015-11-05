
#pragma once

#include <pipeline/common/Grid.h>

#include "deepdecoder.h"

namespace deepdecoder {
class GridGenerator;

class GeneratedGrid : public Grid {
public:
    virtual ~GeneratedGrid() override = default;

    int getLabelAsInt() const;

    template<typename Dtype>
    inline std::vector<Dtype> getLabelAsVector() const {
        return triboolIDtoVector<Dtype>(_ID);
    }
    inline std::string getLabelsAsString() const {
        return deepdecoder::getLabelsAsString(_ID);
    }
    inline const std::vector<std::vector<cv::Point>> & getCoordinates2D() const {
        return _coordinates2D;
    }
    GeneratedGrid scale(double factor) const;
    friend class GridGenerator;
protected:
    explicit GeneratedGrid(Grid::idarray_t id, cv::Point2i center, double radius,
                           double angle_x, double angle_y, double angle_z);
private:
    cv::Scalar tribool2Color(const boost::logic::tribool &tribool) const;
};

}
