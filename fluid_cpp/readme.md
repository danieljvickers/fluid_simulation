# Fluid C++

## About

This is the accelerated Navier-Stokes solver, written in C++ and CUDA C++. Both use the same entry point for the code and data structures, but one is a CPU-based implementation while the other is a GPU-based implementation. This code is instended to showcase the speedup of the python Navier-Stoles solver when replaced with C++ and CUDA.

## Building

This code generates a shared object (`.so`) library for C++, to be included in your personal projects. It also inclused a timing script that will gather benchmarks for the same parameters as the python code.