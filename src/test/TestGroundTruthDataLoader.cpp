
#define CATCH_CONFIG_MAIN

#include <catch.hpp>
#include <boost/filesystem.hpp>
#include <opencv/highgui.h>
#include "CaffeGridGenerator.h"
#include <GroundTruthDataLoader.h>

using namespace deepdecoder;
namespace io = boost::filesystem;


TEST_CASE("CaffeEvaluator", "") {
    std::vector<std::string> gt_files{
        "testdata/Cam_0_20140804152006_3",
        "testdata/Cam_0_20140804152006_3",
        "testdata/Cam_0_20140804152006_3"
    };
    GroundTruthDataLoader<float> data_loader(gt_files);
    size_t batch_size = 64;
    SECTION("it returns images and labels as batches") {
        io::remove_all("gt_loader_images");
        io::create_directories("gt_loader_images");
        dataset_t<float> batch = data_loader.batch(batch_size).get();
        REQUIRE(batch.first.size() == batch_size);
        REQUIRE(batch.second.size() == batch_size);

        for(size_t i = 0; i < batch.first.size(); i++) {
            auto & mat = batch.first.at(i);
            auto & label = batch.second.at(i);
            std::stringstream ss;
            ss << "gt_loader_images/";
            for(size_t j = 0; j < label.size(); j++) {
                ss << label.at(j) << '_';
            }
            ss << ".jpeg";
            cv::imwrite(ss.str(), mat);
        }
    }
}


