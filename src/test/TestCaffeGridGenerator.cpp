
#define CATCH_CONFIG_MAIN

#include <catch.hpp>
#include <opencv/highgui.h>
#include "CaffeGridGenerator.h"
#include <chrono>


using namespace deepdecoder;
TEST_CASE("It adds the geneator to the network and can perform one step", "") {
    caffe::SGDSolver<float> solver("testdata/test_caffe_trainer_solver.prototxt");
    addGridGeneratorToMemoryDataLayer(solver);
    solver.Step(1);
}


