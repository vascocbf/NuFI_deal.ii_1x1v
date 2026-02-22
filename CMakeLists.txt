##
#  CMake script for NuFI FEM Poisson Prototype
##

cmake_minimum_required(VERSION 3.13.4)

# Name of your executable (must match your .cc filename!)
set(TARGET "nufi_poisson")

# List your source files here
set(TARGET_SRC
  ${TARGET}.cc
)

find_package(deal.II 9.6.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR
    "\n*** Could not locate deal.II. ***\n"
    "Pass -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
  )
endif()

deal_ii_initialize_cached_variables()

project(${TARGET})

add_executable(${TARGET} ${TARGET_SRC})

deal_ii_setup_target(${TARGET})
