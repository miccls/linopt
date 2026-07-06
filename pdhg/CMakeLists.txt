cmake_minimum_required(VERSION 3.25)
project(pdhg LANGUAGES CXX)

option(PDHG_BUILD_TESTS "Build GoogleTest tests" ON)
option(PDHG_BUILD_LINT_TARGET "Build the clang-tidy lint target" ON)
option(PDHG_BUILD_FORMAT_TARGETS "Build the clang-format targets" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)

FetchContent_Declare(
  metal_cpp
  GIT_REPOSITORY https://github.com/apple/metal-cpp.git
  GIT_TAG 27c4382b7151d55a51692cdcb27aaa98752240de
)
FetchContent_Declare(
  magic_enum
  GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
  GIT_TAG v0.9.8
)
FetchContent_Declare(
  ms_gsl
  GIT_REPOSITORY https://github.com/microsoft/GSL.git
  GIT_TAG v4.2.0
)
if(POLICY CMP0169)
  cmake_policy(PUSH)
  cmake_policy(SET CMP0169 OLD)
endif()
FetchContent_GetProperties(metal_cpp)
if(NOT metal_cpp_POPULATED)
  FetchContent_Populate(metal_cpp)
endif()
FetchContent_GetProperties(magic_enum)
if(NOT magic_enum_POPULATED)
  FetchContent_Populate(magic_enum)
endif()
FetchContent_GetProperties(ms_gsl)
if(NOT ms_gsl_POPULATED)
  FetchContent_Populate(ms_gsl)
endif()
if(POLICY CMP0169)
  cmake_policy(POP)
endif()

file(GLOB PDHG_SOURCES CONFIGURE_DEPENDS
  src/*.cpp
)

set(PDHG_LINT_SOURCES
  ${PDHG_SOURCES}
)
set(PDHG_FORMAT_SOURCES
  ${PDHG_SOURCES}
  src/linear_algebra.h
  src/metal_details.h
)

add_library(pdhg_core
  ${PDHG_SOURCES}
)
target_include_directories(pdhg_core
  PUBLIC
    src
    ${metal_cpp_SOURCE_DIR}
    ${magic_enum_SOURCE_DIR}/include
    ${ms_gsl_SOURCE_DIR}/include
)
target_link_libraries(pdhg_core
  PRIVATE
    "-framework Foundation"
    "-framework QuartzCore"
    "-framework Metal"
)

if(PDHG_BUILD_TESTS)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_executable(pdhg_tests
    tests/test_linear_algebra.cpp
    tests/test_metal_details.cpp
  )
  list(APPEND PDHG_LINT_SOURCES
    tests/test_linear_algebra.cpp
    tests/test_metal_details.cpp
  )
  list(APPEND PDHG_FORMAT_SOURCES
    tests/test_linear_algebra.cpp
    tests/test_metal_details.cpp
  )
  target_link_libraries(pdhg_tests PRIVATE pdhg_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(pdhg_tests)
endif()

if(PDHG_BUILD_FORMAT_TARGETS)
  find_program(CLANG_FORMAT_EXE NAMES clang-format)
  if(CLANG_FORMAT_EXE)
    add_custom_target(format
      COMMAND ${CLANG_FORMAT_EXE}
              -i
              ${PDHG_FORMAT_SOURCES}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Formatting C++ sources with clang-format"
      VERBATIM
    )
    add_custom_target(format-check
      COMMAND ${CLANG_FORMAT_EXE}
              --dry-run
              --Werror
              ${PDHG_FORMAT_SOURCES}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Checking C++ source formatting with clang-format"
      VERBATIM
    )
  else()
    add_custom_target(format
      COMMAND ${CMAKE_COMMAND} -E echo
              "clang-format was not found. Install clang-format to format C++ sources."
      COMMAND ${CMAKE_COMMAND} -E false
      VERBATIM
    )
    add_custom_target(format-check
      COMMAND ${CMAKE_COMMAND} -E echo
              "clang-format was not found. Install clang-format to check C++ formatting."
      COMMAND ${CMAKE_COMMAND} -E false
      VERBATIM
    )
  endif()
endif()

if(PDHG_BUILD_LINT_TARGET)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    add_custom_target(lint
      COMMAND ${CLANG_TIDY_EXE}
              --quiet
              -p ${CMAKE_BINARY_DIR}
              ${PDHG_LINT_SOURCES}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Running clang-tidy naming checks"
      VERBATIM
    )
  else()
    add_custom_target(lint
      COMMAND ${CMAKE_COMMAND} -E echo
              "clang-tidy was not found. Install LLVM/clang-tidy to run naming checks."
      COMMAND ${CMAKE_COMMAND} -E false
      VERBATIM
    )
  endif()
endif()
