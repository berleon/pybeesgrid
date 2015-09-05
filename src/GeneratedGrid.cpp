
#include "GeneratedGrid.h"
#include <caffe/util/io.hpp>
namespace deepdecoder {

GridGenerator::GridGenerator() : _re(long(time(0))) {
    setAngle(0., 2*M_PI*(60./360.));
    setZAngle(0., 2*M_PI);
    setWhite(0x80, 0xa0);
    setBlack(0x20, 0x40);
    setBackground(0x38, 0x48);
    setGaussianBlur(2, 8);
    _coin_dis = std::bernoulli_distribution(0.5);
}

Grid::idarray_t GridGenerator::generateID() {
    Grid::idarray_t ID;
    for (size_t i = 0; i < ID.size(); i++) {
        ID[i] = _coin_dis(_re);
    }
    return ID;
}

GeneratedGrid GridGenerator::randomGrid()  {
    int b = _black_dis(_re);
    int w = _white_dis(_re);
    Grid::idarray_t id = generateID();
    cv::Scalar black = cv::Scalar(b, b, b);
    cv::Scalar white = cv::Scalar(w, w, w);
    double angle_x = _angle_dis(_re);
    double angle_y = _angle_dis(_re);
    double angle_z = _z_angle_dis(_re);
    long background_color = _background_dis(_re);
    double gaussian_blur = _gaussian_blur_dis(_re);
    return GeneratedGrid(id, black, white, angle_x, angle_y, angle_z,
                         background_color, gaussian_blur);
}

GeneratedGrid::GeneratedGrid(Grid::idarray_t id, cv::Scalar black, cv::Scalar white,
                             double angle_x, double angle_y, double angle_z,
                             long background_color, double gaussian_blur) :
        Grid(cv::Point2i(0, 0), RADIUS, angle_z, angle_y, angle_x), _black(black), _white(white),
         _background_color(background_color), _gaussian_blur(gaussian_blur)
{
    _ID = id;
    prepare_visualization_data();
}

GeneratedGrid::~GeneratedGrid() = default;


std::vector<cv::Point> translate(const std::vector<cv::Point> points, const cv::Point &offset) {
    std::vector<cv::Point> translated_pts;
    translated_pts.reserve(points.size());
    for(const auto & p : points) {
        translated_pts.push_back(p + offset);
    }
    return translated_pts;
}
/**
 * draws the tag at position center in image dst
 *
 * @param img dst
 * @param center center of the tag
 */
void GeneratedGrid::draw(cv::Mat &img, const cv::Point &center) const {
    const auto outer_white_ring = translate(_coordinates2D.at(INDEX_OUTER_WHITE_RING), center);
    const auto inner_white_semicircle = translate(_coordinates2D.at(INDEX_INNER_WHITE_SEMICIRCLE), center);
    const auto inner_black_semicircle = translate(_coordinates2D.at(INDEX_INNER_BLACK_SEMICIRCLE), center);

    cv::fillConvexPoly(img, outer_white_ring, _white);
    for (size_t i = INDEX_MIDDLE_CELLS_BEGIN; i < INDEX_MIDDLE_CELLS_BEGIN + NUM_MIDDLE_CELLS; ++i)
    {
        const auto cell = translate(_coordinates2D.at(i), center);
        cv::Scalar color = tribool2Color(_ID[i - INDEX_MIDDLE_CELLS_BEGIN]);
        cv::fillConvexPoly(img, cell, color);
    }
    cv::fillConvexPoly(img, inner_white_semicircle, _white);
    cv::fillConvexPoly(img, inner_black_semicircle, _black);
    cv::GaussianBlur(img, img, cv::Size(_gaussian_blur_ks, _gaussian_blur_ks), _gaussian_blur);
}

cv::Scalar GeneratedGrid::tribool2Color(const boost::logic::tribool &tribool) const
{
    int value = 0;
    switch (tribool.value) {
        case boost::logic::tribool::value_t::true_value:
            return _white;
        case boost::logic::tribool::value_t::indeterminate_value:
            value = static_cast<int>(0.5 * 255);
            break;
        case boost::logic::tribool::value_t::false_value:
            return _black;
        default:
            assert(false);
            value = 0;
            break;
    }
    return cv::Scalar(value, value, value);
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
    for(size_t i = 0; i < batch_size; i++) {
        GeneratedGrid grid = gen.randomGrid();
        cv::Mat mat(cv::Size(TAG_SIZE, TAG_SIZE), type);
        grid.draw(mat, cv::Point(TAG_SIZE/2, TAG_SIZE/2));
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

cv::Mat GeneratedGrid::cvMat() const {
    cv::Mat mat(TAG_SIZE, TAG_SIZE, CV_8U, _background_color);
    draw(mat, cv::Point2i(TAG_SIZE/2, TAG_SIZE/2));
    return mat;
}
}

