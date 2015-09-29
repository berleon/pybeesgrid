
#pragma once

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include <boost/logic/tribool.hpp>

#include <bits/unique_ptr.h>
#include <caffe/proto/caffe.pb.h>
#include <biotracker/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/common/Grid.h>

#include "deepdecoder.h"

namespace deepdecoder {
class GridGenerator;

struct RBF {
    cv::Point center;
    long intensity;
    cv::Point2f std;
    double correlation;

    double operator()(long x, long y) {
        const double x_term = (x - center.x) / std.x;
        const double y_term = (y - center.y) / std.y;
        double p_ = correlation;
        double exponent = - 0.5/ (1 - std::pow(p_, 2)) *
                (std::pow(x_term, 2) + std::pow(y_term, 2) + 2*p_ * x_term * y_term);
        return intensity * 0.5 / (M_PI * std.x * std.y * sqrt(1 - std::pow(p_, 2))) * exp(exponent);
    }
};

struct GridBackground {
    long background_color;
    std::vector<RBF> rbfs;
};

class GeneratedGrid : public Grid {
public:
    static const size_t RADIUS = 25;
    virtual ~GeneratedGrid() override;

    int getLabelAsInt() const;

    template<typename Dtype>
    std::vector<Dtype> getLabelAsVector() const {
        return triboolIDtoVector<Dtype>(_ID);
    }
    inline std::string getLabelsAsString() const {
        return deepdecoder::getLabelsAsString(_ID);
    }
    inline const std::vector<std::vector<cv::Point>> & getCoordinates2D() const {
        return _coordinates2D;
    }

    friend class GridGenerator;
protected:
    explicit GeneratedGrid(cv::Point2i center, Grid::idarray_t id,
                           double angle_x, double angle_y, double angle_z);
private:
    cv::Scalar tribool2Color(const boost::logic::tribool &tribool) const;
};

#define DISTRIBUTION_MEMBER(NAME, DISTRIBUTION_T, TYPE) \
public:\
    inline void set##NAME(TYPE begin, TYPE end) { \
        _##NAME = std::make_pair(begin, end); \
        _##NAME##_dis = DISTRIBUTION_T(begin, end); \
    } \
    inline std::pair<TYPE, TYPE> get##NAME() const { \
        return _##NAME; \
    }; \
private: \
    std::pair<TYPE, TYPE> _##NAME; \
    DISTRIBUTION_T _##NAME##_dis;

#define UNIFORM_INT_DISTRIBUTION_MEMBER(NAME) DISTRIBUTION_MEMBER( NAME, std::uniform_int_distribution<>, long)
#define UNIFORM_REAL_DISTRIBUTION_MEMBER(NAME) DISTRIBUTION_MEMBER(NAME, std::uniform_real_distribution<>, double)
#define NORMAL_DISTRIBUTION_MEMBER(NAME) DISTRIBUTION_MEMBER(NAME, std::normal_distribution<>, double)

class GridGenerator {
public:
    GridGenerator();
    GeneratedGrid randomGrid();
    UNIFORM_REAL_DISTRIBUTION_MEMBER(YawAngle)
    UNIFORM_REAL_DISTRIBUTION_MEMBER(PitchAngle)
    UNIFORM_REAL_DISTRIBUTION_MEMBER(RollAngle)
    NORMAL_DISTRIBUTION_MEMBER(Center)
private:
    Grid::idarray_t generateID();
    std::mt19937_64 _re;
    std::bernoulli_distribution _coin_dis;
};



class GridArtist {
public:
    void inline draw(const GeneratedGrid & grid, cv::Mat & img) {
        cv::Point2i center(img.rows/2, img.cols/2);
        this->_draw(grid, img, center);
    }
    inline void draw(const GeneratedGrid & grid, cv::Mat & img, cv::Point2i center) {
        this->_draw(grid, img, center);
    }

    cv::Mat draw(const GeneratedGrid & grid);
    virtual ~GridArtist() = default;
protected:
    virtual void _draw(const GeneratedGrid & grid, cv::Mat & img, cv::Point2i center) = 0;
};

class BlackWhiteArtist : public GridArtist {
public:
    virtual ~BlackWhiteArtist() = default;
protected:
    virtual void _draw(const GeneratedGrid & grid, cv::Mat & img, cv::Point2i center);
};

class BadGridArtist : public GridArtist {
    UNIFORM_REAL_DISTRIBUTION_MEMBER(GaussianBlurStd)
    UNIFORM_INT_DISTRIBUTION_MEMBER(White)
    UNIFORM_INT_DISTRIBUTION_MEMBER(Black)
    UNIFORM_INT_DISTRIBUTION_MEMBER(Background)
public:
    BadGridArtist();
    virtual ~BadGridArtist() = default;
protected:
    virtual void _draw(const GeneratedGrid & grid, cv::Mat & img, cv::Point2i center);
private:
    std::mt19937_64 _re;
    cv::Scalar pickColorForTribool(const boost::logic::tribool &tribool, int black, int white) const;
};

class MaskGridArtist : public GridArtist{
public:
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

    virtual ~MaskGridArtist() = default;
protected:
    virtual void _draw(const GeneratedGrid & grid, cv::Mat & img, cv::Point2i center);
private:
    unsigned char maskForTribool(size_t cell_idx, boost::logic::tribool cell_value);
};
}
