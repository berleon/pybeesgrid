
#define CATCH_CONFIG_MAIN

#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>

#include "../GeneratedGrid.h"
#include "../GridArtist.h"
#include <chrono>
#include <catch.hpp>

// uncomment to show window with generated tags
#define VISUAL_TEST

using namespace beesgrid;
void drawGridsOnCvMat(
        const std::vector<GeneratedGrid> & grids,
        size_t size,
        cv::Mat & mat) {
    size_t rows = 10;
    size_t cols = 10;
    MaskGridArtist artist;
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
    size_t size = 64;
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
    SECTION("MaskArtist") {
        MaskGridArtist artist;
        GridGenerator gen;
        const GeneratedGrid grid = gen.randomGrid();
        cv::Mat mat(size, size, CV_8UC1);
        artist.draw(grid, mat, cv::Point2i(size/2, size/2));
        std::array<size_t, MASK_LEN> counts;
        std::fill(counts.begin(), counts.end(), 0);
        const size_t white_offset = 13;
        for(auto p = mat.datastart; p != mat.dataend; p++) {
            for(size_t i = 0; i < MASK_LEN; i++) {
                if(MASK_INDICIES[i] == *p) {
                    counts.at(i) += 1;
                }
            }
        }
        for(size_t i = 1; i < Grid::NUM_MIDDLE_CELLS + 1; i++) {
            INFO("i:" << i << ", i+offset: " << i + IGNORE << ", counts[i]: "
                 << counts[i] << ", counts[i+offset]:" << counts.at(i + white_offset));
            CHECK((counts.at(i) > 0 || counts.at(i + white_offset) > 0));
        }
        for(size_t i = 0; i < MASK_LEN; i++) {
            if ((i >= 1 && i < 1 + Grid::NUM_MIDDLE_CELLS) ||
                (i >= white_offset + 1 && i < 1 + white_offset + Grid::NUM_MIDDLE_CELLS)) {
                continue;
            }
            CHECK(counts.at(i) > 0);
        }
    }

    SECTION("BlackWhiteArtist") {
        u_int8_t black = 51;
        u_int8_t white = 255;
        u_int8_t background = 0;
        double antialiasing = 4;
        BlackWhiteArtist artist(black, white, background, antialiasing);
        GridGenerator gen;
        const GeneratedGrid grid = gen.randomGrid();
        cv::Mat mat(size, size, CV_8UC1);
        artist.draw(grid, mat, cv::Point2i(size/2, size/2));
        size_t nb_blacks = 0;
        size_t nb_whites = 0;
        size_t nb_backgrounds = 0;
        for(size_t x = 0; x < size; x++) {
            for(size_t y = 0; y < size; y++) {
                auto pixel = mat.at<u_int8_t>(x, y);
                if (pixel == black) {
                    nb_blacks++;
                } else if(pixel == white) {
                    nb_whites++;
                } else if(pixel == background) {
                    nb_backgrounds++;
                }
            }
        }
        cv::imwrite("blackwhite.png", mat);
        REQUIRE(nb_blacks > 0);
        REQUIRE(nb_whites > 0);
        REQUIRE(nb_backgrounds > 0);
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

