file(GLOB src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
file(GLOB hdr RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)

add_library(deepdecoder STATIC ${src} ${hdr} )

target_link_libraries(deepdecoder ${libs})

set(test-libs deepdecoder ${libs})

add_executable(cv_exercise_data cv_exercise_data.cpp)
target_link_libraries(cv_exercise_data deepdecoder)

add_subdirectory(test)
