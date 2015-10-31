
#include "GeneratedGrid.h"
#include <caffe/util/io.hpp>
namespace deepdecoder {

constexpr double to_radian(double degrees) {
    return (M_PI / 180) * degrees;
}
GridGenerator::GridGenerator() : _re(long(time(0)))
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
GeneratedGrid::GeneratedGrid(Grid::idarray_t id, cv::Point2i center, double radius,
                             double angle_x, double angle_y, double angle_z) :
        Grid(center, radius, angle_z, angle_y, angle_x)
{
    _ID = id;
    prepare_visualization_data();
}



std::vector<cv::Point> translate(const std::vector<cv::Point> points, const cv::Point &offset) {
    std::vector<cv::Point> translated_pts;
    translated_pts.reserve(points.size());
    for(const auto & p : points) {
        translated_pts.push_back(p + offset);
    }
    return translated_pts;
}


caffe::Datum gridToCaffeDatum(const GeneratedGrid & grid, const cv::Mat & mat) {
    caffe::Datum datum;
    std::vector<uchar> buf;
    cv::imencode(".jpeg", mat, buf);
    datum.set_channels(mat.channels());
    datum.set_width(TAG_SIZE);
    datum.set_height(TAG_SIZE);
    datum.set_data(std::string(reinterpret_cast<char *>(&buf[0]),
                               buf.size()));
    datum.set_label(grid.getLabelAsInt());
    datum.set_encoded(true);
    return datum;
}

std::vector<caffe::Datum> generateData(size_t batch_size, GridGenerator & gen, bool greyscale) {
    int type = greyscale ? CV_8U : CV_8UC3;
    std::vector<caffe::Datum> data(batch_size);
    BadGridArtist drawer;
    for(size_t i = 0; i < batch_size; i++) {
        GeneratedGrid grid = gen.randomGrid();
        cv::Mat mat(cv::Size(TAG_SIZE, TAG_SIZE), type);
        drawer.draw(grid, mat, cv::Point(TAG_SIZE/2, TAG_SIZE/2));
        data.emplace_back(gridToCaffeDatum(grid, mat));
    }
    return data;
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


BadGridArtist::BadGridArtist()
{
    setWhite(0x40, 0x60);
    setBlack(0x1F, 0x30);
    setBackground(0x38, 0x48);
    setGaussianBlurStd(1.2, 1.8);
}


void MaskGridArtist::_draw(const GeneratedGrid &grid, cv::Mat &img, cv::Point2i center)
{
    auto & coords2d = grid.getCoordinates2D();
    img.setTo(MASK::IGNORE);
    const auto outer_white_ring = translate(coords2d.at(Grid::INDEX_OUTER_WHITE_RING), center);
    const auto inner_white_semicircle = translate(coords2d.at(Grid::INDEX_INNER_WHITE_SEMICIRCLE), center);
    const auto inner_black_semicircle = translate(coords2d.at(Grid::INDEX_INNER_BLACK_SEMICIRCLE), center);
    const auto background_ring = translate(coords2d.at(Grid::INDEX_INNER_BLACK_SEMICIRCLE), center);

    cv::fillConvexPoly(img, background_ring, MASK::BACKGROUND_RING);
    cv::fillConvexPoly(img, outer_white_ring, MASK::OUTER_WHITE_RING);
    const auto max_i = Grid::INDEX_MIDDLE_CELLS_BEGIN + Grid::NUM_MIDDLE_CELLS;
    for (size_t i = Grid::INDEX_MIDDLE_CELLS_BEGIN; i < max_i; ++i)
    {
        const auto cell = translate(coords2d.at(i), center);
        const size_t id_idx = i - static_cast<long>(Grid::INDEX_MIDDLE_CELLS_BEGIN);
        cv::Scalar color = MaskGridArtist::maskForTribool(id_idx,  grid.getIdArray().at(id_idx));
        cv::fillConvexPoly(img, cell, color);
    }
    cv::fillConvexPoly(img, inner_white_semicircle, MASK::INNER_WHITE_SEMICIRCLE);
    cv::fillConvexPoly(img, inner_black_semicircle, MASK::INNER_BLACK_SEMICIRCLE);
}

unsigned char MaskGridArtist::maskForTribool(size_t cell_idx, boost::logic::tribool cell_value)
{
    if(cell_value) {
        return MASK::CELL_0_WHITE + cell_idx;
    } else if(!cell_value) {
        return MASK::CELL_0_BLACK + cell_idx;
    } else {
        throw "GridMaskDrawer cannot handle indeterminate value";
    }
}

cv::Scalar BadGridArtist::pickColorForTribool(const boost::logic::tribool &tribool,
                                              int black, int white) const
{
    int value = 0;
    if(tribool) {
        value = white;
    } else if(!tribool) {
        value = black;
    } else {
        throw "pickColorForTribool cannot handle indeterminate value.";
    }
    return cv::Scalar(value, value, value);
}

void BadGridArtist::_draw(const GeneratedGrid &grid, cv::Mat &img, cv::Point2i center)
{

    int black = _Black_dis(_re);
    int white = _White_dis(_re);
    const auto & coords2D = grid.getCoordinates2D();
    const auto outer_white_ring = translate(coords2D.at(Grid::INDEX_OUTER_WHITE_RING), center);
    const auto inner_white_semicircle = translate(coords2D.at(Grid::INDEX_INNER_WHITE_SEMICIRCLE), center);
    const auto inner_black_semicircle = translate(coords2D.at(Grid::INDEX_INNER_BLACK_SEMICIRCLE), center);

    cv::fillConvexPoly(img, outer_white_ring, white);
    for (size_t i = Grid::INDEX_MIDDLE_CELLS_BEGIN; i < Grid::INDEX_MIDDLE_CELLS_BEGIN + Grid::NUM_MIDDLE_CELLS; ++i)
    {
        const auto cell = translate(coords2D.at(i), center);
        cv::Scalar color = BadGridArtist::pickColorForTribool(
                    grid.getIdArray()[i - Grid::INDEX_MIDDLE_CELLS_BEGIN], black, white);
        cv::fillConvexPoly(img, cell, color);
    }
    cv::fillConvexPoly(img, inner_white_semicircle, white);
    cv::fillConvexPoly(img, inner_black_semicircle, black);
}

cv::Mat GridArtist::draw(const GeneratedGrid &grid)
{
    cv::Mat mat(TAG_SIZE, TAG_SIZE, CV_8U, cv::Scalar(0));
    this->draw(grid, mat, TAG_CENTER);
    return mat;
}

void BlackWhiteArtist::_draw(const GeneratedGrid &grid, cv::Mat &img, cv::Point2i center)
{
    static int black = 0;
    static int white = 255;
    const auto & coords2D = grid.getCoordinates2D();
    const auto outer_white_ring = translate(coords2D.at(Grid::INDEX_OUTER_WHITE_RING), center);
    const auto inner_white_semicircle = translate(coords2D.at(Grid::INDEX_INNER_WHITE_SEMICIRCLE), center);
    const auto inner_black_semicircle = translate(coords2D.at(Grid::INDEX_INNER_BLACK_SEMICIRCLE), center);
    const auto background_ring = translate(coords2D.at(Grid::INDEX_BACKGROUND_RING), center);
    cv::fillConvexPoly(img, background_ring, 128);
    cv::fillConvexPoly(img, outer_white_ring, white);
    for (size_t i = Grid::INDEX_MIDDLE_CELLS_BEGIN; i < Grid::INDEX_MIDDLE_CELLS_BEGIN + Grid::NUM_MIDDLE_CELLS; ++i)
    {
        const auto cell = translate(coords2D.at(i), center);
        boost::tribool bit = grid.getIdArray()[i - Grid::INDEX_MIDDLE_CELLS_BEGIN];
        cv::Scalar color = bit ? white : black;
        cv::fillConvexPoly(img, cell, color);
    }
    cv::fillConvexPoly(img, inner_white_semicircle, white);
    cv::fillConvexPoly(img, inner_black_semicircle, black);
}

}
