file(GLOB src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
file(GLOB hdr RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)
list(REMOVE_ITEM src "main.cpp")

add_library(deepdecoder ${src} ${hdr} )

target_link_libraries(deepdecoder ${libs})

add_executable(main "main.cpp"
     ${hdr} ${UI_RESOURCES} ${UI_HEADERS})

set(test-libs deepdecoder ${libs})
add_subdirectory(test)
