
#define CATCH_CONFIG_MAIN

#include <catch.hpp>
#include <boost/filesystem.hpp>
#include <opencv/highgui.h>
#include "CaffeGridGenerator.h"
#include <CaffeEvaluator.h>

using namespace deepdecoder;
namespace io = boost::filesystem;


TEST_CASE("CaffeEvaluator", "") {
    std::vector<std::string> gt_files{
            "testdata/Cam_0_20140804152006_3",
            "testdata/Cam_0_20140804152006_3",
            "testdata/Cam_0_20140804152006_3"
    };
    CaffeEvaluator evaluator(std::move(gt_files));
    SECTION("it can read all ground truth images") {
        io::create_directories("gt_images");
        dataset_t<float> dataset = evaluator.getAllData<float>();
        REQUIRE(dataset.first.size() > 100);
        for(size_t i = 0; i < dataset.first.size(); i++) {
            auto & mat = dataset.first.at(i);
            auto output_path = io::unique_path("gt_images/%%%%%%%%%%%.jpeg");
            cv::imwrite(output_path.string(), mat);
        }
    }
    SECTION("it evaluate a netork") {
        caffe::Net<float> net("testdata/test_caffe_evaluator_network.prototxt", caffe::TEST);
        auto result = evaluator.evaluate(net);
        REQUIRE(result.accuracy < 0.01);
    }
}


