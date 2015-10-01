cmake_minimum_required(VERSION 2.8.7)

cmake_policy(SET CMP0020 NEW)
cmake_policy(SET CMP0009 NEW)
cmake_policy(SET CMP0012 NEW)
cmake_policy(SET CMP0046 NEW)

project(deepdecoder)

#------------------------------------------------------------------------------
# Required CPM Setup - no need to modify - See: https://github.com/iauns/cpm
#------------------------------------------------------------------------------
set(CPM_DIR "${CMAKE_CURRENT_BINARY_DIR}/cpm_packages" CACHE TYPE STRING)
find_package(Git)
if(NOT GIT_FOUND)
    message(FATAL_ERROR "CPM requires Git.")
endif()
if (NOT EXISTS ${CPM_DIR}/CPM.cmake)
    message(STATUS "Cloning repo (https://github.com/iauns/cpm)")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" clone https://github.com/iauns/cpm ${CPM_DIR}
      RESULT_VARIABLE error_code
      OUTPUT_QUIET ERROR_QUIET)
    if(error_code)
        message(FATAL_ERROR "CPM failed to get the hash for HEAD")
    endif()
endif()
include(${CPM_DIR}/CPM.cmake)


if(NOT DEFINED CMAKECONFIG_PATH)
    CPM_AddModule("cmakeconfig"
        GIT_REPOSITORY "https://github.com/BioroboticsLab/cmakeconfig.git"
        GIT_TAG "master")
else()
    CPM_AddModule("cmakeconfig"
        SOURCE_DIR "${CMAKECONFIG_PATH}")
endif()

set(CAFFE_COMMIT "285cad9a74d5ab85596369c17b93262780aacdf7")
set(CAFFE_GIT_REPO "https://github.com/berleon/caffe")
set(USE_DEEPLOCALIZER 1)
set(BLAS Open CACHE STRING "Use openblas")
set(BUILD_python Off)
set(BUILD_python_layer Off)
set(BIOTRACKER_BUILD_TESTS OFF CACHE BOOL "Don't build biotrackers tests")

include_pipeline("1c05b839536ce530f89453fe329247ce1b2b96dc")
include_biotracker_core("606e6144615a1efda634278efa06497a0bc88681")

CPM_AddModule("cpm_catch"
  GIT_REPOSITORY "https://github.com/bcachet/cpm-catch")

CPM_Finish()

biorobotics_config()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ ${CMAKE_MODULE_PATH})

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(Boost COMPONENTS system filesystem serialization python3 program_options REQUIRED)
find_package(LMDB REQUIRED)
find_package(CUDA REQUIRED)
find_package(gflags REQUIRED)


include_directories(
    SYSTEM ${CATCH_INCLUDE_DIR}
    SYSTEM ${OpenCV_INCLUDE_DIRS}
    SYSTEM ${LMDB_INCLUDE_DIRS}
    SYSTEM ${CUDA_INCLUDE_DIRS}
    SYSTEM ${DEEP_CLASSIFIER_INCLUDE_DIR}
)

set(libs
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${PYTHON_LIBRARIES}
    ${LMDB_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CPM_LIBRARIES}
)


enable_testing(true)
add_subdirectory(src)
add_subdirectory(python)
