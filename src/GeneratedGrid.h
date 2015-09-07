
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
    /**
     * draws 2D projection of 3D-mesh on image
     */
    void draw(cv::Mat &img, const cv::Point& center) const;
    cv::Mat cvMat() const;
    friend class GridGenerator;
protected:
    explicit GeneratedGrid(cv::Point2i center, Grid::idarray_t id, cv::Scalar black, cv::Scalar white,
                           double angle_x, double angle_y, double angle_z,
                           double gaussian_blur, GridBackground background);
private:
    static const size_t _gaussian_blur_ks = 7;
    cv::Scalar _black;
    cv::Scalar _white;
    double _gaussian_blur;
    GridBackground _background;
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

    inline void setReflectionSpotProb(const double prob) {
        _reflection_spot_prob = prob;
        _reflection_spot_dis = std::bernoulli_distribution(prob);
    }

    GeneratedGrid randomGrid();
    UNIFORM_REAL_DISTRIBUTION_MEMBER(YawAngle)
    UNIFORM_REAL_DISTRIBUTION_MEMBER(PitchAngle)
    UNIFORM_REAL_DISTRIBUTION_MEMBER(RollAngle)
    UNIFORM_REAL_DISTRIBUTION_MEMBER(GaussianBlurStd)
    UNIFORM_INT_DISTRIBUTION_MEMBER(White)
    UNIFORM_INT_DISTRIBUTION_MEMBER(Black)
    UNIFORM_INT_DISTRIBUTION_MEMBER(Background)
    NORMAL_DISTRIBUTION_MEMBER(Center)
private:
    Grid::idarray_t generateID();
    double _reflection_spot_prob;


    std::mt19937_64 _re;
    std::bernoulli_distribution _reflection_spot_dis;

    std::bernoulli_distribution _coin_dis;
};

std::vector<caffe::Datum> generateData(size_t batch_size, GridGenerator & gen,  bool greyscale = true);

}
