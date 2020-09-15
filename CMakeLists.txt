##
#  CMake script
##

# Set the name of the project and target:
SET(TARGET "cracks")

SET(TARGET_SRC
  ${TARGET}.cc
  )

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 9.0.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

IF( NOT DEAL_II_WITH_MPI OR
    NOT DEAL_II_WITH_P4EST OR
    NOT DEAL_II_WITH_TRILINOS )
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_MPI = ON
    DEAL_II_WITH_P4EST = ON
    DEAL_II_WITH_TRILINOS = ON
One or all of these are OFF in your installation but are required for this program."
    )
ENDIF()


DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()

# define SOURCE_DIR:

FOREACH(_source_file ${TARGET_SRC})
  SET_PROPERTY(SOURCE ${_source_file}
    APPEND PROPERTY COMPILE_DEFINITIONS SOURCE_DIR="${CMAKE_SOURCE_DIR}")
ENDFOREACH()


# indenting:

ADD_CUSTOM_TARGET(indent
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND ./contrib/indent
  COMMENT "Indenting all header and source files..."
  )


# testing:

ENABLE_TESTING()

ADD_SUBDIRECTORY(tests)
