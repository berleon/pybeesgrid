
file(GLOB src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

set(PYTHON_PKG_DIR ${CMAKE_CURRENT_BINARY_DIR}/package)

add_library(pydeepdecoder SHARED ${src})

target_include_directories(pydeepdecoder SYSTEM
    PUBLIC "${CMAKE_SOURCE_DIR}/src"
    PUBLIC ${PYTHON_INCLUDE_DIRS}
)
message(${PYTHON_LIBRARIES})
target_link_libraries(pydeepdecoder
        deepdecoder
)
set_target_properties(pydeepdecoder PROPERTIES PREFIX "" )

add_custom_target(create_python_pkg)
add_dependencies(create_python_pkg pydeepdecoder)
add_custom_command(TARGET create_python_pkg POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} ${PYTHON_PKG_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/pydeepdecoder.so ${PYTHON_PKG_DIR}/deepdecoder/)
