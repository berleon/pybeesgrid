

#include <opencv/highgui.h>
#include "GeneratedGrid.h"
#include "GridArtist.h"
#include "GridGenerator.h"

void noise(cv::Mat &mat);

using namespace beesgrid;

void noise(cv::Mat & mat) {
    std::random_device rd;
    std::bernoulli_distribution coin(0.1);

    for(int x = 0; x < mat.rows; x++) {
        for(int y = 0; y < mat.cols; y++) {
            if (coin(rd)) {
                if (mat.at<uchar>(x, y) == 0) {
                    mat.at<uchar>(x, y) = 255;
                } else {
                    mat.at<uchar>(x, y) = 0;
                }
            }
        }
    }
}

int main() {
    size_t size = 64;
    GridGenerator gen;
    const size_t nb_grids = 3;
    BlackWhiteArtist artist;
    for(size_t i = 0; i < nb_grids; i++) {
        GeneratedGrid grid = gen.randomGrid();
        {
            cv::Mat mat(size, size, CV_8UC1, cv::Scalar(0));
            std::stringstream ss;
            ss << "tag" << i << "_all_random";
            artist.draw(grid, mat);
            cv::imwrite(ss.str() + ".png", mat);
            noise(mat);
            ss << "_noise.png";
            cv::imwrite(ss.str(), mat);
        }

        grid.setXRotation(0);
        grid.setZRotation(0);
        {
            cv::Mat mat(size, size, CV_8UC1, cv::Scalar(0));
            std::stringstream ss;
            ss << "tag" << i << "_xy_random";
            artist.draw(grid, mat);
            cv::imwrite(ss.str() + ".png", mat);
            noise(mat);
            ss << "_noise.png";
            cv::imwrite(ss.str(), mat);
        }
        grid.setYRotation(0);
        {
            cv::Mat mat(size, size, CV_8UC1, cv::Scalar(0));
            std::stringstream ss;
            ss << "tag" << i << "_zero";
            artist.draw(grid, mat);
            cv::imwrite(ss.str() + ".png", mat);
            noise(mat);
            ss << "_noise.png";
            cv::imwrite(ss.str(), mat);
        }
    }
}

