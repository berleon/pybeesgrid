# Find the native numpy includes
# This module defines
#  NUMPY_INCLUDE_DIR, where to find numpy/arrayobject.h, etc.
#  NUMPY_FOUND

# Find Python
execute_process(
    COMMAND
    which python
    OUTPUT_VARIABLE PYTHON_EXECUTABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT NUMPY_INCLUDE_DIR)
    exec_program ("${PYTHON_EXECUTABLE}"
      ARGS "-c 'import numpy; print(numpy.get_include())'"
      OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
      RETURN_VALUE NUMPY_FOUND)
    if (NUMPY_FOUND)
      if (NOT NUMPY_FIND_QUIETLY)
        message(STATUS "Numpy headers found")
        mark_as_advanced(NUMPY_INCLUDE_DIR)
      endif (NOT NUMPY_FIND_QUIETLY)
    else (NUMPY_FOUND)
      if (NUMPY_FIND_REQUIRED)
        message (FATAL_ERROR "Numpy headers missing")
      endif (NUMPY_FIND_REQUIRED)
    endif (NUMPY_FOUND)
endif (NOT NUMPY_INCLUDE_DIR)
