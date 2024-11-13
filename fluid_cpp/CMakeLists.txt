cmake_minimum_required(VERSION 3.28)
project(fluid_cpp)

set(CMAKE_CXX_STANDARD 17)

add_library(fluid_cpp SHARED
        cpp/NavierStokesSolver.cpp
        cpp/NavierStokesSolver.h)

add_executable(main main.cpp)
target_link_libraries(main fluid_cpp)
