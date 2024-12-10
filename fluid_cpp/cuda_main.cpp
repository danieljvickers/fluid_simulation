//
// Created by dan on 12/10/24.
//

#include <iostream>
#include <chrono>
#include <fstream>
#include "cuda/ParallelNavierStokes.cuh"

int main() {
    // set up the solver
    int num_x_bins = 41;
    int num_y_bins = 41;
    float width = 1.0;
    float height = 1.0;
    ParallelNavierStokes<float> solver(num_x_bins, num_y_bins, width, height);

    // entire specific constants of the simulation
    solver.density = 1.0;
    solver.kinematic_viscosity = 0.1;
    solver.num_iterations = 1000;
    solver.num_poisson_iterations = 50;
    solver.time_step = 0.001;
    solver.stability_safety_factor = 0.5;

    std::cout << "Hello World" << std::endl;
    /*

    // establish boundary conditions
    for (int x = 0; x < num_x_bins; x++) {
        solver.setUBoundaryCondition(x, 0, 0.);  // no flow inside of the floor
        solver.setVBoundaryCondition(x, 0, 0.);  // no flow into the floor
        solver.setUBoundaryCondition(x, num_y_bins - 1, 1.);  // water flowing to the right on top
        solver.setVBoundaryCondition(x, num_y_bins - 1, 0.);  // no flow into the top
        solver.setPBoundaryCondition(x, num_y_bins - 1, 0.);  // no pressure at the top
    }
    for (int y = 1; y < num_y_bins - 1; y++) {
        solver.setUBoundaryCondition(0, y, 0.);  // no flow into of the left wall
        solver.setVBoundaryCondition(0, y, 0.);  // no flow inside the left wall
        solver.setUBoundaryCondition(num_x_bins - 1, y, 0.);  // no floow into the right wall
        solver.setVBoundaryCondition(num_x_bins - 1, y, 0.);  // no flow inside the right wall
    }

    // Solve and retrieve the solution
    solver.solve();
    auto* u_values = static_cast<float*>(malloc(sizeof(float) * num_x_bins * num_y_bins));
    auto* v_values = static_cast<float*>(malloc(sizeof(float) * num_x_bins * num_y_bins));
    auto* p_values = static_cast<float*>(malloc(sizeof(float) * num_x_bins * num_y_bins));
    solver.getUValues(u_values);
    solver.getVValues(v_values);
    solver.getPValues(p_values);*/
}