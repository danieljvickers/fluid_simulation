//
// Created by dan on 12/17/24.
//

#include <iostream>
#include <chrono>
#include <fstream>

#include "cuda_runtime.h"
#include "../cpp/SerialNavierStokes.h"
#include "../cuda/ParallelNavierStokes.cuh"

int main() {
    // set up the solver
    int num_x_bins = 41;
    int num_y_bins = 41;
    float width = 1.0;
    float height = 1.0;
    SerialNavierStokes<float> unit_solver(num_x_bins, num_y_bins, width, height);
    ParallelNavierStokes<float> tested_solver(num_x_bins, num_y_bins, width, height);

    // entire specific constants of the simulation
    unit_solver.density = 1.0;
    unit_solver.kinematic_viscosity = 0.1;
    unit_solver.num_iterations = 1000;
    unit_solver.num_poisson_iterations = 50;
    unit_solver.time_step = 0.001;
    unit_solver.stability_safety_factor = 0.5;

    tested_solver.density = unit_solver.density;
    tested_solver.kinematic_viscosity = unit_solver.kinematic_viscosity;
    tested_solver.num_iterations = unit_solver.num_iterations;
    tested_solver.num_poisson_iterations = unit_solver.num_poisson_iterations;
    tested_solver.time_step = unit_solver.time_step;
    tested_solver.stability_safety_factor = unit_solver.stability_safety_factor;

    // establish boundary conditions
    for (int x = 0; x < num_x_bins; x++) {
        unit_solver.setUBoundaryCondition(x, 0, 0.);  // no flow inside of the floor
        unit_solver.setVBoundaryCondition(x, 0, 0.);  // no flow into the floor
        unit_solver.setUBoundaryCondition(x, num_y_bins - 1, 1.);  // water flowing to the right on top
        unit_solver.setVBoundaryCondition(x, num_y_bins - 1, 0.);  // no flow into the top
        unit_solver.setPBoundaryCondition(x, num_y_bins - 1, 0.);  // no pressure at the top

        tested_solver.setUBoundaryCondition(x, 0, 0.);  // no flow inside of the floor
        tested_solver.setVBoundaryCondition(x, 0, 0.);  // no flow into the floor
        tested_solver.setUBoundaryCondition(x, num_y_bins - 1, 1.);  // water flowing to the right on top
        tested_solver.setVBoundaryCondition(x, num_y_bins - 1, 0.);  // no flow into the top
        tested_solver.setPBoundaryCondition(x, num_y_bins - 1, 0.);  // no pressure at the top
    }
    for (int y = 1; y < num_y_bins - 1; y++) {
        unit_solver.setUBoundaryCondition(0, y, 0.);  // no flow into of the left wall
        unit_solver.setVBoundaryCondition(0, y, 0.);  // no flow inside the left wall
        unit_solver.setUBoundaryCondition(num_x_bins - 1, y, 0.);  // no floow into the right wall
        unit_solver.setVBoundaryCondition(num_x_bins - 1, y, 0.);  // no flow inside the right wall

        tested_solver.setUBoundaryCondition(0, y, 0.);  // no flow into of the left wall
        tested_solver.setVBoundaryCondition(0, y, 0.);  // no flow inside the left wall
        tested_solver.setUBoundaryCondition(num_x_bins - 1, y, 0.);  // no floow into the right wall
        tested_solver.setVBoundaryCondition(num_x_bins - 1, y, 0.);  // no flow inside the right wall
    }

    // Solve and retrieve the solution
    unit_solver.solve();
    tested_solver.migrateSolve();

    auto* unit_values = static_cast<float*>(malloc(sizeof(float) * num_x_bins * num_y_bins));
    unit_solver.getVValues(unit_values);

    auto* tested_values = static_cast<float*>(malloc(sizeof(float) * num_x_bins * num_y_bins));
    tested_solver.getVValues(tested_values);

    int num_errors = 0;
    for (int x = 0; x < num_x_bins; x++) {
        for (int y = 0; y < num_y_bins; y++) {
            int ind = x * num_y_bins + y;
            if (unit_values[ind] == tested_values[ind]) {
                std::cout << "SUCCESS ";
            } else {
                std::cout << "FAILURE ";
                num_errors++;
            }
            std::cout << unit_values[ind] << " :: " << tested_values[ind] << " x=" << x << " y="<< y <<  std::endl;
        }
    }

    std::cout << "Passed " << 100.f * float((num_x_bins * num_y_bins) - num_errors) / float(num_x_bins * num_y_bins) << "% cells" << std::endl;

    // clean up
    free(unit_values);
    free(tested_values);

    return 0;
}
