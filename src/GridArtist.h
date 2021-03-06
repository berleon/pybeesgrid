#pragma once

#include "GeneratedGrid.h"

#include "GridGenerator.h"

namespace beesgrid {

class GridArtist {
public:
    void inline draw(const GeneratedGrid &grid, cv::Mat &img) {
        cv::Point2i image_center(img.rows / 2, img.cols / 2);
        this->draw(grid, img, image_center);
    }

    inline void draw(const GeneratedGrid &grid, cv::Mat &img, cv::Point2i offset) {
        this->_draw(grid, img, offset + grid.getCenter());
    }

    virtual std::unique_ptr <GridArtist> clone() const = 0;

    cv::Mat draw(const GeneratedGrid &grid);

    virtual ~GridArtist() = default;

protected:
    virtual void _draw(const GeneratedGrid &grid, cv::Mat &img, cv::Point2i center) = 0;
};

class DepthMapArtist : public GridArtist {
public:
    virtual std::unique_ptr<GridArtist> clone() const override {
        return std::make_unique<DepthMapArtist>();
    }

    virtual ~DepthMapArtist() override = default;
protected:
    virtual void _draw(const GeneratedGrid &grid, cv::Mat &img, cv::Point2i center);
    static constexpr double DISTANCE_BETWEEN_POINTS = 0.5;
    static constexpr double DISTANCE_BETWEEN_RINGS = 0.5;
};

class BlackWhiteArtist : public GridArtist {
public:
    BlackWhiteArtist(u_int8_t black=0, u_int8_t white=255,
                     u_int8_t background=0, double antialiasing=1);
    virtual ~BlackWhiteArtist() = default;

    virtual std::unique_ptr <GridArtist> clone() const {
        return std::make_unique<BlackWhiteArtist>(_black,
                                                  _white, _background, _antialiasing);
    }

protected:
    u_int8_t _white;
    u_int8_t _black;
    u_int8_t _background;
    double _antialiasing;
    virtual void _draw(const GeneratedGrid &grid, cv::Mat &img, cv::Point2i center);
};

class BadGridArtist : public GridArtist {
    UNIFORM_REAL_DISTRIBUTION_MEMBER(GaussianBlurStd)
    UNIFORM_INT_DISTRIBUTION_MEMBER(White)
    UNIFORM_INT_DISTRIBUTION_MEMBER(Black)
    UNIFORM_INT_DISTRIBUTION_MEMBER(Background)
public:
    BadGridArtist();

    virtual std::unique_ptr <GridArtist> clone() const {
        return std::make_unique<BadGridArtist>();
    }

    virtual ~BadGridArtist() = default;

protected:
    virtual void _draw(const GeneratedGrid &grid, cv::Mat &img, cv::Point2i center);

private:
    std::mt19937_64 _re;

    cv::Scalar pickColorForTribool(const boost::logic::tribool &tribool, int black, int white) const;
};

const static size_t MASK_LEN = 28;

enum MASK {
    INNER_BLACK_SEMICIRCLE,
    CELL_0_BLACK = 1,
    CELL_1_BLACK,
    CELL_2_BLACK,
    CELL_3_BLACK,
    CELL_4_BLACK,
    CELL_5_BLACK,
    CELL_6_BLACK,
    CELL_7_BLACK,
    CELL_8_BLACK,
    CELL_9_BLACK,
    CELL_10_BLACK,
    CELL_11_BLACK,
    IGNORE = 128,
    CELL_0_WHITE = IGNORE + 1,
    CELL_1_WHITE = IGNORE + 2,
    CELL_2_WHITE = IGNORE + 3,
    CELL_3_WHITE = IGNORE + 4,
    CELL_4_WHITE = IGNORE + 5,
    CELL_5_WHITE = IGNORE + 6,
    CELL_6_WHITE = IGNORE + 7,
    CELL_7_WHITE = IGNORE + 8,
    CELL_8_WHITE = IGNORE + 9,
    CELL_9_WHITE = IGNORE + 10,
    CELL_10_WHITE = IGNORE + 11,
    CELL_11_WHITE = IGNORE + 12,
    OUTER_WHITE_RING = IGNORE + 20,
    INNER_WHITE_SEMICIRCLE = IGNORE + 21
};
const unsigned char MASK_INDICIES[] = {
        INNER_BLACK_SEMICIRCLE, CELL_0_BLACK, CELL_1_BLACK, CELL_2_BLACK,
        CELL_3_BLACK, CELL_4_BLACK, CELL_5_BLACK, CELL_6_BLACK, CELL_7_BLACK,
        CELL_8_BLACK, CELL_9_BLACK, CELL_10_BLACK, CELL_11_BLACK, IGNORE,
        CELL_0_WHITE, CELL_1_WHITE, CELL_2_WHITE, CELL_3_WHITE, CELL_4_WHITE,
        CELL_5_WHITE, CELL_6_WHITE, CELL_7_WHITE, CELL_8_WHITE, CELL_9_WHITE,
        CELL_10_WHITE, CELL_11_WHITE, OUTER_WHITE_RING, INNER_WHITE_SEMICIRCLE
};

class MaskGridArtist : public GridArtist {
public:
    virtual ~MaskGridArtist() = default;

    virtual std::unique_ptr <GridArtist> clone() const {
        return std::make_unique<MaskGridArtist>();
    }

protected:
    virtual void _draw(const GeneratedGrid &grid, cv::Mat &img, cv::Point2i center);

private:
    unsigned char maskForTribool(size_t cell_idx, boost::logic::tribool cell_value) const;
};

}
