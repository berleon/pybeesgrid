
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
    SECTION("RandomGridDrawer") {
        cv::Mat big_mat(cv::Size(size*cols, size*rows), CV_8UC3, cv::Scalar(0x40, 0x40, 0x40));
        GridGenerator gen;
        BadGridArtist artist;
        for(size_t i = 0; i < rows; i++) {
            for(size_t j = 0; j < cols; j++) {
                GeneratedGrid grid = gen.randomGrid();
                cv::Mat submat = big_mat.rowRange(i*size, (i+1)*size).colRange(j*size, (j+1)*size);
                artist.draw(grid, submat, cv::Point2i(size/2, size/2));
                cv::putText(submat, grid.getLabelsAsString(), cv::Point(5, 55),
                            cv::FONT_HERSHEY_SIMPLEX, 0.2, cv::Scalar(255));
            }
        }
        # ifdef VISUAL_TEST
        cv::namedWindow("window", cv::WINDOW_AUTOSIZE);
        cv::imshow("window", big_mat);
        cv::waitKey(0);
        # endif
        cv::imwrite("generated_tags.jpeg", big_mat);
    }


    SECTION("bench RandomGridDrawer") {
        GridGenerator gen;
        BadGridArtist artist;
        size_t loops = 20;
        auto start = std::chrono::system_clock::now();
        for(size_t i = 0; i < loops; i++) {
            for(size_t b = 0; b < 64; b++) {
                GeneratedGrid gg = gen.randomGrid();
                artist.draw(gg);
            }

        }
        std::chrono::duration<double> time_per_loop = (std::chrono::system_clock::now() - start) / loops;
        std::cout << "time per loop: " << time_per_loop.count() << std::endl;
    }

}
