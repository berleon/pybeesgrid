
message( ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/source)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/source)

add_custom_target(python DEPENDS deepdecoder)

add_custom_command(TARGET python POST_BUILD
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/source/"
    COMMAND ${CMAKE_COMMAND} -E env
    "LDFLAGS=\"-L${CMAKE_BINARY_DIR}/src\""
    python setup.py build)

add_custom_target(python_pip_install DEPENDS deepdecoder)

add_custom_command(TARGET python_pip_install POST_BUILD
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/source/"
    COMMAND ${CMAKE_COMMAND} -E env
    "LDFLAGS=\"-L${CMAKE_BINARY_DIR}/src\""
    pip install . --upgrade)

