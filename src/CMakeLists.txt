file(GLOB src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
file(GLOB hdr RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)
list(REMOVE_ITEM src "main.cpp"  "caffe.cpp")

add_library(deepdecoder STATIC ${src} ${hdr} )

target_link_libraries(deepdecoder ${libs})

add_executable(main "main.cpp"
     ${hdr} ${UI_RESOURCES} ${UI_HEADERS})

add_executable(caffe-gen.bin "caffe.cpp"
        ${hdr} ${UI_RESOURCES} ${UI_HEADERS})

set_target_properties(caffe-gen.bin PROPERTIES
        COMPILE_FLAGS "-Wno-error=sign-compare -Wno-error=pedantic"
)
add_definitions(-DCPU_ONLY)
target_link_libraries(caffe-gen.bin deepdecoder gflags)

set(test-libs deepdecoder ${libs})
add_subdirectory(test)
