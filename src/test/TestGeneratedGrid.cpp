
#define CATCH_CONFIG_MAIN

#include <catch.hpp>
#include <opencv/highgui.h>
#include "../GeneratedGrid.h"
#include <chrono>

// uncomment to show window with generated tags
// #define VISUAL_TEST

using namespace deepdecoder;
void drawGridsOnCvMat(
        const std::vector<GeneratedGrid> & grids,
        size_t size,
        cv::Mat & mat) {
    size_t rows = 10;
    size_t cols = 10;
    BlackWhiteArtist artist;
    assert(rows*cols == grids.size());
    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            const GeneratedGrid & grid = grids.at(rows*i + j);
            cv::Mat submat = mat.rowRange(i*size, (i+1)*size).colRange(j*size, (j+1)*size);
            artist.draw(grid, submat, cv::Point2i(size/2, size/2));
        }
    }
}

TEST_CASE("Grid is generated and drawn", "") {
    size_t size = 60;
    size_t rows = 10;
    size_t cols = 10;
    SECTION("RandomGridDrawer") {
        cv::Mat big_mat(cv::Size(size*cols, size*rows), CV_8UC1, cv::Scalar(0));
        GridGenerator gen;
        std::vector<GeneratedGrid> grids;
        double scale = 1./4;
        std::vector<GeneratedGrid> scaled_grids;
        for(size_t i = 0; i < rows*cols; i++) {
            GeneratedGrid grid = gen.randomGrid();
            scaled_grids.emplace_back(grid.scale(scale));
            grids.emplace_back(std::move(grid));
        }
        drawGridsOnCvMat(grids, size, big_mat);
        cv::imwrite("generated_tags.jpeg", big_mat);
        # ifdef VISUAL_TEST
        cv::namedWindow("window", cv::WINDOW_AUTOSIZE);
        cv::imshow("window", big_mat);
        cv::waitKey(0);
        # endif

        size_t scaled_size = size_t(size * scale);
        cv::Mat mat_scaled(cv::Size(scaled_size*cols, scaled_size*rows),
                           CV_8UC1, cv::Scalar(0));

        drawGridsOnCvMat(scaled_grids, scaled_size, mat_scaled);
        cv::imwrite("generated_tags_scaled.jpeg", mat_scaled);
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

