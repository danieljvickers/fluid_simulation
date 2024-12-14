cmake_minimum_required(VERSION 3.28)
project(fluid_cpp)

set(CMAKE_CXX_STANDARD 17)
find_package(CUDAToolkit)
enable_language(CUDA)

add_library(fluid_cpp SHARED
        cpp/NavierStokesSolver.cpp
        cpp/NavierStokesSolver.h
        cpp/NavierStokesCell.h
        cpp/SerialNavierStokes.cpp
        cpp/SerialNavierStokes.h
)

add_library(fluid_cuda SHARED
        cpp/NavierStokesCell.h
        cpp/NavierStokesSolver.cpp
        cpp/NavierStokesSolver.h
        cuda/ParallelNavierStokes.cu
        cuda/ParallelNavierStokes.cuh
)

add_executable(main main.cpp)
target_link_libraries(main fluid_cpp)

add_executable(cuda_main cuda_main.cu)
target_link_libraries(cuda_main fluid_cuda)
