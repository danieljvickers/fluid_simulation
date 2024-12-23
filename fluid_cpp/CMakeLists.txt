cmake_minimum_required(VERSION 3.28)
project(fluid_cpp)

set(CMAKE_CXX_STANDARD 20)
find_package(CUDAToolkit)
enable_language(CUDA)

add_library(fluid_cpp SHARED
        cpp/NavierStokesSolver.cpp
        cpp/NavierStokesSolver.h
        cpp/NavierStokesCell.h
        cpp/SerialNavierStokes.cpp
        cpp/SerialNavierStokes.h
        cuda/ParallelNavierStokes.cu
        cuda/ParallelNavierStokes.cuh
        cpp/ThreadedNavierStokes.cpp
        cpp/ThreadedNavierStokes.h
)

add_executable(main main.cpp)
target_link_libraries(main fluid_cpp)

add_executable(cuda_main cuda_main.cu)
target_link_libraries(cuda_main fluid_cpp)

add_executable(test_main tests/test_main.cpp)
target_link_libraries(test_main fluid_cpp)
