
#pragma once

#include <vector>                  // std::vector
#include <array>                   // std::array
#include <opencv2/opencv.hpp>      // cv::Mat, cv::Point3_
#include <boost/logic/tribool.hpp> // boost::tribool
#include <common/Grid.h>
#include <bits/unique_ptr.h>
#include <caffe/proto/caffe.pb.h>

namespace deepdecoder {

static const size_t TAG_SIZE = 60;
class GeneratedGrid : public Grid {
public:
    // default constructor, required for serialization
    explicit GeneratedGrid();
    explicit GeneratedGrid(cv::Point2i center, double radius, double angle_z,
                           double angle_y, double angle_x);

    virtual ~GeneratedGrid() override;

    int getLabel() const;
    /**
     * draws 2D projection of 3D-mesh on image
     */
    void draw(cv::Mat &img, const cv::Point& center) const;

    void generateView();
    void generateID();
private:
    thread_local static std::mt19937_64 _re;
    static std::uniform_real_distribution<> _angle_dis;
    static std::uniform_int_distribution<>  _white_dis;
    static std::uniform_int_distribution<>  _outer_ring_color_dis;
    static std::uniform_int_distribution<>  _black_dis;
    static std::uniform_real_distribution<> _gaussian_blur_dis;
    static std::bernoulli_distribution _coin_dis;
    cv::Scalar _black;
    cv::Scalar _white;
    cv::Scalar _outer_ring_color;
    double _gaussian_blur;
    cv::Scalar tribool2Color(const boost::logic::tribool &tribool) const;
};
std::vector<caffe::Datum> generateData(size_t batch_size, bool greyscale = true);
}
