cmake_minimum_required(VERSION 2.8.8)
project(pipeline_build CXX)
include(ExternalProject)
find_package(Git REQUIRED)
ExternalProject_Add(
    pipeline
    PREFIX ${CMAKE_BINARY_DIR}/pipeline
    GIT_REPOSITORY /home/leon/uni/vision_swp/pipeline
    # git@maserati.mi.fu-berlin.de:pipeline
    GIT_TAG deeplocalizer
    TIMEOUT 10
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
)

ExternalProject_Get_Property(pipeline BINARY_DIR)
ExternalProject_Get_Property(pipeline SOURCE_DIR)
set(PIPELINE_INCLUDE_DIR ${SOURCE_DIR}/ CACHE INTERNAL "Path to include folder for Pipeline")
set(PIPELINE_LD_DIR ${BINARY_DIR}/ CACHE INTERNAL "Path to library folder for Catch")

