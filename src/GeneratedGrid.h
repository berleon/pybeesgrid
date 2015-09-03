
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
class GridGenerator {
    GridGenerator();

    inline void setWhite(int begin, int end) {
        _white = std::make_pair(begin, end);
    }

    inline void setBlack(int begin, int end) {
        _black = std::make_pair(begin, end);
    }

    inline void setBackground(int begin, int end) {
        _background = std::make_pair(begin, end);
        _background_dis = std::uniform_int_distribution<>(begin, end);
    }

    inline void setGaussianBlur(double begin, double end) {
        _gaussian_blur = std::make_pair(begin, end);
        _gaussian_blur_dis = std::uniform_real_distribution<>(begin, end);
    }
private:
    std::pair<double, double> _z_angle;
    std::pair<double, double> _angle;
    std::pair<double, double> _gaussian_blur;
    std::pair<int, int> _white;
    std::pair<int, int> _black;
    std::pair<int, int> _background;

    std::mt19937_64 _re;
    std::uniform_real_distribution<> _angle_dis; //(0., 2*M_PI*(60./360.));
    std::uniform_real_distribution<> _z_angle_dis; //(0., 2*M_PI);
    std::uniform_int_distribution<>  _white_dis; //(0x80, 0xa0);
    std::uniform_int_distribution<>  _black_dis; //(0x20, 0x40);
    std::uniform_int_distribution<>  _background_dis; //(0x38, 0x48);
    std::uniform_real_distribution<> _gaussian_blur_dis; //(2, 8);
};


class GeneratedGrid : public Grid {
public:
    // default constructor, required for serialization
    explicit GeneratedGrid();
    explicit GeneratedGrid(cv::Point2i center, double radius, double angle_z,
                           double angle_y, double angle_x);

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
    void generateView();
    void generateID();
private:
    thread_local static std::mt19937_64 _re;
    static std::uniform_real_distribution<> _angle_dis;
    static std::uniform_real_distribution<> _z_angle_dis;
    static std::uniform_int_distribution<>  _white_dis;
    static std::uniform_int_distribution<>  _black_dis;
    static std::uniform_real_distribution<> _gaussian_blur_dis;
    static std::uniform_int_distribution<>  _background_dis;
    static std::bernoulli_distribution _coin_dis;
    static const size_t _gaussian_blur_ks = 7;
    cv::Scalar _black;
    cv::Scalar _white;
    cv::Scalar _background_color;
    double _gaussian_blur;
    cv::Scalar tribool2Color(const boost::logic::tribool &tribool) const;
};
std::vector<caffe::Datum> generateData(size_t batch_size, bool greyscale = true);
}
