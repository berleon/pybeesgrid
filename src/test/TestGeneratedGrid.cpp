
#define CATCH_CONFIG_MAIN

#include <catch.hpp>
#include <opencv/highgui.h>
#include "../GeneratedGrid.h"
#include <chrono>

// uncomment to show window with generated tags
// #define VISUAL_TEST

using namespace deepdecoder;
TEST_CASE("Grid is generated and drawn", "") {
    size_t size = 60;
    size_t rows = 10;
    size_t cols = 10;
    cv::Mat big_mat(cv::Size(size*cols, size*rows), CV_8UC3, cv::Scalar(0x40, 0x40, 0x40));
    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            GeneratedGrid grid;
            cv::Mat submat = big_mat.rowRange(i*size, (i+1)*size).colRange(j*size, (j+1)*size);
            grid.draw(submat, cv::Point2i(size/2, size/2));
        }
    }
    # ifdef VISUAL_TEST
    cv::namedWindow("window", cv::WINDOW_AUTOSIZE);
    cv::imshow("window", big_mat);
    cv::waitKey(0);
    # endif
    cv::imwrite("generated_tags.jpeg", big_mat);

    size_t loops = 20;
    auto start = std::chrono::system_clock::now();
    for(size_t i = 0; i < loops; i++) {
        generateData(64);
    }
    std::chrono::duration<double> time_per_loop = (std::chrono::system_clock::now() - start) / loops;
    std::cout << "time per loop: " << time_per_loop.count() << std::endl;

}

