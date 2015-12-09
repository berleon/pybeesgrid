
#define CATCH_CONFIG_MAIN

#include <catch.hpp>
#include <boost/filesystem.hpp>
#include <opencv/highgui.h>
#include <GroundTruthDataLoader.h>

using namespace beesgrid;
namespace io = boost::filesystem;


TEST_CASE("GroundTrouthDataLoader", "") {
    std::vector<std::string> gt_files{
        "testdata/Cam_0_20140804152006_3.tdat",
        "testdata/Cam_0_20140804152006_3.tdat",
        "testdata/Cam_0_20140804152006_3.tdat"
    };
    GroundTruthDataLoader<float> data_loader(gt_files);
    size_t rows = 10;
    size_t cols = 10;
    size_t batch_size = rows*cols;
    SECTION("it returns images and labels as batches") {
        io::remove_all("gt_loader_images");
        io::create_directories("gt_loader_images");
        gt_dataset_t batch = data_loader.batch(batch_size).get();
        REQUIRE(batch.first.size() == batch_size);
        REQUIRE(batch.second.size() == batch_size);
        cv::Mat big_image(rows*TAG_SIZE+rows-1, cols*TAG_SIZE+rows-1,
                          CV_8U, cv::Scalar(0xff));
        for(size_t r = 0; r < rows; r++) {
            for(size_t c = 0; c < cols; c++) {
                size_t i = r*rows + c;
                auto & mat = batch.first.at(i);
                auto & bits = batch.second.at(i).bits;
                std::stringstream ss;
                ss << "gt_loader_images/";
                for(size_t j = 0; j < bits.size(); j++) {
                    ss << bits.at(j) << '_';
                }
                ss << ".png";
                cv::imwrite(ss.str(), mat);
                size_t r_start = r*TAG_SIZE + r;
                size_t c_start = c*TAG_SIZE + c;
                cv::Mat subimage = big_image.rowRange(r_start, r_start+TAG_SIZE)
                        .colRange(c_start, c_start + TAG_SIZE);
                mat.copyTo(subimage);
            }
        }
        cv::imwrite("ground_truth_images.png", big_image);
    }
    SECTION("it can return all images and labels avialable") {
        size_t rows = 25;
        size_t cols = 25;
        cv::Mat big_image(rows*TAG_SIZE+rows-1, cols*TAG_SIZE+rows-1,
                          CV_8U, cv::Scalar(0x0));

        size_t i = 0;
        size_t r = 0;
        size_t c = 0;
        while(auto batch = data_loader.batch(batch_size)) {
            for(const auto & mat : batch.get().first) {
                if(i % 2 == 0 && r < rows) {
                    size_t r_start = r*TAG_SIZE + r;
                    size_t c_start = c*TAG_SIZE + c;
                    cv::Mat subimage = big_image.rowRange(r_start, r_start+TAG_SIZE)
                            .colRange(c_start, c_start + TAG_SIZE);
                    mat.copyTo(subimage);
                    c++;
                    if(c == cols) {
                        r++;
                        c = 0;
                    }
                }
                i++;
            }
        }
        std::cout << i << std::endl;
        cv::imwrite("ground_truth_images_big.png", big_image);
        REQUIRE(not data_loader.batch(batch_size));
    }
}


