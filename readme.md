# Fluid Simulation
Author: Daniel J. Vickers

## About

This repository has been created for me to explore fluid simulation, as an attempt to move towards magnetohydro dynamics simulations. I have opted to start by solving the Navier-Stokes equations for fluid flow before adding the complexity of electromagnetism. This is also an opportunity for me to demonstrate an ability to write software in Python, C++, and CUDA.

All of the python code is written to leverage performance from numpy. This is intended to be the unit test for the higher-performance implementations of the simulation and a performance baseline.

The C++ code is meant to be a serial implementation of Navier-Stokes.

The CUDA code is a large-scale parallel implementation of Navier-Stokes. I am leaving several versions of the simulation, and the data presented here is generated using an NVIDIA RTX 2080 Super, and will not likely not yield similar performance on your machine.

## Python

The most-basic implementation is a python version that leverages linear algebra in numpy. Because we will be implementing the Cpp and CUDA versions raw (without external libraries), the python implementation is quite optimial.

The result of the sim is the quiver plot below that captures the fluid velocity and pressure.

![screenshot](figures/python_solution.png)

I then tested the performance of our Navier-Stokes solver. The parameters here are those that match all solutions in this repo:
- 41x41 bins in our grid
- 1000 time steps forward
- 50 poison steps to let the fluid settle

The results of 5000 time trials for the Python version is shown below.

![screenshot](screenshot.png)