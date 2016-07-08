file(GLOB src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
file(GLOB hdr RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)

add_library(beesgrid STATIC ${src} ${hdr} )

target_link_libraries(beesgrid ${libs})

set(test-libs beesgrid ${libs})

add_subdirectory(test)
