
#include "GridGenerator.h"
namespace deepdecoder {

constexpr double to_radian(double degrees) {
    return (M_PI / 180) * degrees;
}

GridGenerator::GridGenerator() : GridGenerator(static_cast<unsigned long>(time(0))) {}

GridGenerator::GridGenerator(unsigned long seed)  : _re(seed)
{
    setYawAngle(0., 2 * M_PI);
    setPitchAngle(to_radian(-30), to_radian(65));
    setRollAngle(to_radian(-30), to_radian(30));
    setRadius(22, 25);
    setCenter(0, 3);
}

Grid::idarray_t GridGenerator::generateID() {
    Grid::idarray_t ID;
    for (size_t i = 0; i < ID.size(); i++) {
        ID[i] = _coin_dis(_re);
    }
    return ID;
}

std::unique_ptr<GridGenerator> GridGenerator::clone() {
    std::unique_ptr<GridGenerator> gen = std::make_unique<GridGenerator>(static_cast<unsigned long>(_re()));
    gen->setCenter(_Center.first, _Center.second);
    gen->setPitchAngle(_PitchAngle.first, _PitchAngle.second);
    gen->setRollAngle(_RollAngle.first, _RollAngle.second);
    gen->setYawAngle(_YawAngle.first, _YawAngle.second);
    gen->setRadius(_Radius.first, _Radius.second);
    return gen;
}

GeneratedGrid GridGenerator::randomGrid()  {
    Grid::idarray_t id = generateID();
    double angle_z = _YawAngle_dis(_re);
    double angle_y = _PitchAngle_dis(_re);
    double angle_x = _RollAngle_dis(_re);
    size_t radius = _Radius_dis(_re);
    auto center_cords = [&]() { return int(round(_Center_dis(_re))); };
    cv::Point2i center{center_cords(), center_cords()};
    return GeneratedGrid(id, center, radius, angle_x, angle_y, angle_z);
}
}
