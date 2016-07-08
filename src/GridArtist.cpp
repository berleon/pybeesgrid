#include "GridArtist.h"

#include "Grid.h"
#include <boost/logic/tribool.hpp>
#include "util/CvHelper.h"

namespace beesgrid {

std::vector<cv::Point> translate(const std::vector<cv::Point> points, const cv::Point &offset) {
    std::vector<cv::Point> translated_pts;
    translated_pts.reserve(points.size());
    for(const auto & p : points) {
        translated_pts.push_back(p + offset);
    }
    return translated_pts;
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

unsigned char MaskGridArtist::maskForTribool(size_t cell_idx, boost::logic::tribool cell_value) const
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

BlackWhiteArtist::BlackWhiteArtist(u_int8_t black, u_int8_t white, u_int8_t background,
                                   double antialiasing) :
    _white(white),
    _black(black),
    _background(background),
    _antialiasing(antialiasing)
{
}

void BlackWhiteArtist::_draw(const GeneratedGrid &grid, cv::Mat &img, cv::Point2i center)
{
    auto draw_func = [&](const GeneratedGrid & draw_grid, cv::Mat &draw_mat, cv::Point2i draw_center) {
        draw_mat.setTo(_background);
        const auto & coords2D = draw_grid.getCoordinates2D();

        const auto outer_white_ring = translate(coords2D.at(Grid::INDEX_OUTER_WHITE_RING), draw_center);
        const auto inner_white_semicircle = translate(coords2D.at(Grid::INDEX_INNER_WHITE_SEMICIRCLE), draw_center);
        const auto inner_black_semicircle = translate(coords2D.at(Grid::INDEX_INNER_BLACK_SEMICIRCLE), draw_center);
        cv::fillConvexPoly(draw_mat, outer_white_ring, _white);
        for (size_t i = Grid::INDEX_MIDDLE_CELLS_BEGIN; i < Grid::INDEX_MIDDLE_CELLS_BEGIN + Grid::NUM_MIDDLE_CELLS; ++i)
        {
            const auto cell = translate(coords2D.at(i), draw_center);
            boost::tribool bit = draw_grid.getIdArray()[i - Grid::INDEX_MIDDLE_CELLS_BEGIN];
            cv::Scalar color = bit ? _white : _black;
            cv::fillConvexPoly(draw_mat, cell, color);
        }
        cv::fillConvexPoly(draw_mat, inner_white_semicircle, _white);
        cv::fillConvexPoly(draw_mat, inner_black_semicircle, _black);
    };

    if (_antialiasing != 1) {
        cv::Size anti_size(int(img.size().width * _antialiasing),
                           int(img.size().height * _antialiasing));
        cv::Mat draw_mat(anti_size, CV_8U);
        GeneratedGrid draw_grid = grid.scale(_antialiasing);
        cv::Point2i draw_center (int(center.x*_antialiasing), int(center.y*_antialiasing));
        draw_func(draw_grid, draw_mat, draw_center);

        double sigma = 2*_antialiasing/6;
        int ksize = int(2 * ceil(3 * sigma) + 1);
        cv::Mat blured = draw_mat.clone();
        cv::GaussianBlur(draw_mat, blured, cv::Size(ksize, ksize), sigma, sigma,
                         cv::BORDER_REFLECT101);
        cv::resize(blured, img, img.size());
    } else {
        draw_func(grid, img, center);
    }
}

void DepthMapArtist::_draw(const GeneratedGrid &grid, cv::Mat &img, cv::Point2i center) {
    const cv::Rect img_rect(cv::Point(), img.size());
    const auto rotationMatrix = CvHelper::rotationMatrix(grid.getZRotation(), grid.getYRotation(), grid.getXRotation());
    const size_t nb_rings = static_cast<size_t>(grid.getRadius() / DISTANCE_BETWEEN_RINGS);

    const double bulge_factor = grid.structure()->BULGE_FACTOR;
    const double z_outer = grid.z_coordinate(1, bulge_factor);

    for(size_t i = 0; i < nb_rings; i++) {
        const double radius = i / static_cast<double>(nb_rings);
        const double circumference = 2*M_PI*grid.getRadius()*radius;
        const size_t nb_points = std::max(static_cast<size_t>(circumference / DISTANCE_BETWEEN_POINTS), size_t(1));
        for(size_t j = 0; j < nb_points; j++) {
            const double angle = j * 2*M_PI / nb_points;
            const cv::Point3d p(
                    std::cos(angle)*radius,
                    std::sin(angle)*radius,
                    Grid::z_coordinate(radius, bulge_factor) - z_outer
            );
            const cv::Point3d rot_p = rotationMatrix*p;
            const cv::Point2i projectedPoint = grid.projectPoint(rot_p) + center;
            if(img_rect.contains(projectedPoint)) {
                // z is between [-1, 1], map it to [0, 1] and invert it, so that higher z means nearer
                double z = 1 - rot_p.z / 2 + 0.5;
                u_int8_t z_as_byte = static_cast<u_int8_t >(255 * z);
                u_int8_t  current_z = img.at<u_int8_t>(projectedPoint.y, projectedPoint.x);
                if (z_as_byte > current_z) {
                    img.at<u_int8_t>(projectedPoint.y, projectedPoint.x) = z_as_byte;
                }
            }
        }
    }
}

}
