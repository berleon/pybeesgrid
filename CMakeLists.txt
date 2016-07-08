cmake_minimum_required(VERSION 2.8.7)

cmake_policy(SET CMP0020 NEW)
cmake_policy(SET CMP0009 NEW)
cmake_policy(SET CMP0012 NEW)
cmake_policy(SET CMP0046 NEW)

project(pybeesgrid)

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

CPM_AddModule("cpm_catch"
  GIT_REPOSITORY "https://github.com/bcachet/cpm-catch")

CPM_Finish()

biorobotics_config()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ ${CMAKE_MODULE_PATH})

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost COMPONENTS system filesystem python3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(Numpy REQUIRED)

include_directories(
    SYSTEM ${CATCH_INCLUDE_DIR}
    SYSTEM ${OpenCV_INCLUDE_DIRS}
)

set(libs
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${PYTHON_LIBRARIES}
    ${CPM_LIBRARIES}
)


enable_testing(true)
add_subdirectory(src)
add_subdirectory(python)
