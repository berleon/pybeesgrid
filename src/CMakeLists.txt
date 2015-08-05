file(GLOB_RECURSE src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
file(GLOB_RECURSE hdr RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)
list(REMOVE_ITEM src "main.cpp")

include_directories(${PROJECT_SOURCE_DIR}/include/deeplocalizer/tagger/)

add_library(deepdecoder
    ${src} ${hdr} ${UI_RESOURCES} ${UI_HEADERS})

target_link_libraries(deepdecoder ${libs})

add_executable(main "main.cpp"
     ${hdr} ${UI_RESOURCES} ${UI_HEADERS})

set(test-libs deepdecoder)
add_subdirectory(test)
