
#include "GeneratedGrid.h"
#include <caffe/util/io.hpp>
namespace deepdecoder {

thread_local std::mt19937_64 GeneratedGrid::_re(long(time(0)));
std::uniform_real_distribution<> GeneratedGrid::_angle_dis(0., M_PI*0.25);
std::uniform_int_distribution<>  GeneratedGrid::_white_dis(0x60, 0x80);
std::uniform_int_distribution<>  GeneratedGrid::_outer_ring_color_dis(0x90, 0xb0);
std::uniform_int_distribution<>  GeneratedGrid::_black_dis(0x20, 0x40);
std::uniform_real_distribution<> GeneratedGrid::_gaussian_blur_dis(1, 6);
std::bernoulli_distribution GeneratedGrid::_coin_dis(0.5);
GeneratedGrid::GeneratedGrid()
        : GeneratedGrid(cv::Point2i(0, 0), 25., 0, 0, 0) {
}

GeneratedGrid::GeneratedGrid(cv::Point2i center, double radius_px,
                             double angle_z, double angle_y, double angle_x)
        : Grid(center, radius_px, angle_z, angle_y, angle_x)
{

    generateID();
    generateView();
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
    cv::GaussianBlur(img, img, cv::Size(7, 7), _gaussian_blur);
}

cv::Scalar GeneratedGrid::tribool2Color(const boost::logic::tribool &tribool) const
{
    int value = 0;
    switch (tribool.value) {
        case boost::logic::tribool::value_t::true_value:
            return _black;
        case boost::logic::tribool::value_t::indeterminate_value:
            value = static_cast<int>(0.5 * 255);
            break;
        case boost::logic::tribool::value_t::false_value:
            return _white;
        default:
            assert(false);
            value = 0;
            break;
    }
    return cv::Scalar(value, value, value);
}

void GeneratedGrid::generateView() {
    int b = _black_dis(_re);
    int w = _white_dis(_re);
    int o = _outer_ring_color_dis(_re);
    _black = cv::Scalar(b, b, b);
    _white = cv::Scalar(w, w, w);
    _outer_ring_color = cv::Scalar(o, o, o);
    _angle_x = _angle_dis(_re);
    _angle_y = _angle_dis(_re);
    _angle_z = _angle_dis(_re);
    _gaussian_blur = _gaussian_blur_dis(_re);
}
void GeneratedGrid::generateID() {

    for (size_t i = 0; i < _ID.size(); i++) {
        _ID[i] = _coin_dis(_re);
    }
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
    datum.set_label(grid.getLabel());
    datum.set_encoded(true);
    return datum;
}
std::vector<caffe::Datum> generateData(size_t batch_size, bool greyscale) {
    int type = greyscale ? CV_8U : CV_8UC3;
    std::vector<caffe::Datum> data(batch_size);
    for(size_t i = 0; i < batch_size; i++) {
        GeneratedGrid grid;
        cv::Mat mat(cv::Size(TAG_SIZE, TAG_SIZE), type);
        grid.draw(mat, cv::Point(TAG_SIZE/2, TAG_SIZE/2));
        data.emplace_back(gridToCaffeDatum(grid, mat));
    }
    return data;
}
int GeneratedGrid::getLabel() const {
    // TODO: check how the tag is structured
    int label = 0;
    for(size_t i = 0; i < _ID.size(); i++) {
        if(_ID[i]) {
            label += (1 << i);
        }
    }
    return label;
}
}

