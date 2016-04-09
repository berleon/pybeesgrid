#include "GridArtist.h"

#include <pipeline/common/Grid.h>
#include <boost/logic/tribool.hpp>

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
    const auto background_ring = translate(coords2d.at(Grid::INDEX_BACKGROUND_RING), center);

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

BlackWhiteArtist::BlackWhiteArtist(u_int8_t black, u_int8_t white, u_int8_t background) :
    _white(white),
    _black(black),
    _background(background)
{

}
void BlackWhiteArtist::_draw(const GeneratedGrid &grid, cv::Mat &img, cv::Point2i center)
{
    img.setTo(_background);
    const auto & coords2D = grid.getCoordinates2D();
    const auto outer_white_ring = translate(coords2D.at(Grid::INDEX_OUTER_WHITE_RING), center);
    const auto inner_white_semicircle = translate(coords2D.at(Grid::INDEX_INNER_WHITE_SEMICIRCLE), center);
    const auto inner_black_semicircle = translate(coords2D.at(Grid::INDEX_INNER_BLACK_SEMICIRCLE), center);
    cv::fillConvexPoly(img, outer_white_ring, _white);
    for (size_t i = Grid::INDEX_MIDDLE_CELLS_BEGIN; i < Grid::INDEX_MIDDLE_CELLS_BEGIN + Grid::NUM_MIDDLE_CELLS; ++i)
    {
        const auto cell = translate(coords2D.at(i), center);
        boost::tribool bit = grid.getIdArray()[i - Grid::INDEX_MIDDLE_CELLS_BEGIN];
        cv::Scalar color = bit ? _white : _black;
        cv::fillConvexPoly(img, cell, color);
    }
    cv::fillConvexPoly(img, inner_white_semicircle, _white);
    cv::fillConvexPoly(img, inner_black_semicircle, _black);
}

}
