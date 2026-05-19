cmake_minimum_required(VERSION 3.25)
project(linopt_native LANGUAGES CXX)

option(LINOPT_NATIVE_BUILD_TESTS "Build native GoogleTest tests" OFF)
option(LINOPT_NATIVE_BUILD_PYTHON "Build Python extension module" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)

FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()
add_library(Eigen3::Eigen INTERFACE IMPORTED)
target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})

add_library(linopt_native_core
  cpp/src/dual_simplex.cpp
  cpp/src/ipm_tools.cpp
  cpp/src/pivot_strategy.cpp
  cpp/src/predictor_corrector.cpp
  cpp/src/primal_simplex.cpp
  cpp/src/simplex_utils.cpp
)
target_include_directories(linopt_native_core PUBLIC cpp/include)
target_link_libraries(linopt_native_core PUBLIC Eigen3::Eigen)

if(LINOPT_NATIVE_BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native cpp/src/bindings.cpp)
  target_link_libraries(_native PRIVATE linopt_native_core)
  install(TARGETS _native LIBRARY DESTINATION linopt_native)
endif()

if(LINOPT_NATIVE_BUILD_TESTS)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_executable(linopt_native_tests
    cpp/tests/test_simplex.cpp
    cpp/tests/test_ipm.cpp
  )
  target_link_libraries(linopt_native_tests PRIVATE linopt_native_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(linopt_native_tests)
endif()
