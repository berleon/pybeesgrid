
#include "GeneratedGrid.h"
namespace beesgrid {

GeneratedGrid::GeneratedGrid(Grid::idarray_t id, cv::Point2i center, double radius,
                             double angle_x, double angle_y, double angle_z) :
        Grid(center, radius, angle_z, angle_y, angle_x)
{
    _ID = id;
    prepare_visualization_data();
}





int GeneratedGrid::getLabelAsInt() const {
    // TODO: check how the tag is structured
    int label = 0;
    for(size_t i = 0; i < _ID.size(); i++) {
        if(_ID[i]) {
            label += (1 << i);
        }
    }
    return label;
}

GeneratedGrid GeneratedGrid::scale(double factor) const
{
    return GeneratedGrid(_ID, _center, _radius * factor,
                         _angle_x, _angle_y, _angle_z);
}


}
