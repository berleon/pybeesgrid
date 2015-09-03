
#pragma once

#include <vector>                  // std::vector
#include <array>                   // std::array
#include <opencv2/opencv.hpp>      // cv::Mat, cv::Point3_
#include <boost/logic/tribool.hpp> // boost::tribool

#include <bits/unique_ptr.h>
#include <caffe/proto/caffe.pb.h>
#include <biotracker/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/common/Grid.h>

#include "deepdecoder.h"

namespace deepdecoder {
class GridGenerator;

class GeneratedGrid : public Grid {
public:
    static const size_t RADIUS = 25;
    virtual ~GeneratedGrid() override;

    int getLabelAsInt() const;

    template<typename Dtype>
    std::vector<Dtype> getLabelAsVector() const {
        return triboolIDtoVector<Dtype>(_ID);
    }
    /**
     * draws 2D projection of 3D-mesh on image
     */
    void draw(cv::Mat &img, const cv::Point& center) const;
    cv::Mat cvMat() const;
    friend class GridGenerator;
protected:
    explicit GeneratedGrid(Grid::idarray_t id, cv::Scalar black, cv::Scalar white,
                           double angle_x, double angle_y, double angle_z,
                           long background_color, double gaussian_blur);
private:
    static const size_t _gaussian_blur_ks = 7;
    cv::Scalar _black;
    cv::Scalar _white;
    cv::Scalar _background_color;
    double _gaussian_blur;
    cv::Scalar tribool2Color(const boost::logic::tribool &tribool) const;
};


class GridGenerator {
public:
    GridGenerator();

    inline void setAngle(double begin, double end) {
        _angle = std::make_pair(begin, end);
        _angle_dis = std::uniform_real_distribution<>(begin, end);
    }

    inline void setZAngle(double begin, double end) {
        _z_angle = std::make_pair(begin, end);
        _z_angle_dis = std::uniform_real_distribution<>(begin, end);
    }

    inline void setWhite(int begin, int end) {
        _white = std::make_pair(begin, end);
        _white_dis = std::uniform_int_distribution<>(begin, end);
    }

    inline void setBlack(int begin, int end) {
        _black = std::make_pair(begin, end);
        _black_dis = std::uniform_int_distribution<>(begin, end);
    }

    inline void setBackground(int begin, int end) {
        _background = std::make_pair(begin, end);
        _background_dis = std::uniform_int_distribution<>(begin, end);
    }

    inline void setGaussianBlur(double begin, double end) {
        _gaussian_blur = std::make_pair(begin, end);
        _gaussian_blur_dis = std::uniform_real_distribution<>(begin, end);
    }

    GeneratedGrid randomGrid();
private:
    Grid::idarray_t generateID();
    std::pair<double, double> _z_angle;
    std::pair<double, double> _angle;
    std::pair<double, double> _gaussian_blur;
    std::pair<int, int> _white;
    std::pair<int, int> _black;
    std::pair<int, int> _background;

    std::mt19937_64 _re;
    std::uniform_real_distribution<> _angle_dis;
    std::uniform_real_distribution<> _z_angle_dis;
    std::uniform_int_distribution<>  _white_dis;
    std::uniform_int_distribution<>  _black_dis;
    std::uniform_int_distribution<>  _background_dis;
    std::uniform_real_distribution<> _gaussian_blur_dis;
    std::bernoulli_distribution _coin_dis;
};

std::vector<caffe::Datum> generateData(size_t batch_size, GridGenerator & gen,  bool greyscale = true);

}
