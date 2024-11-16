#include <iostream>
#include <chrono>
#include <fstream>
#include "cpp/NavierStokesSolver.h"

int main() {
    // set up the solver
    int num_x_bins = 41;
    int num_y_bins = 41;
    float width = 1.0;
    float height = 1.0;

    NavierStokesSolver<float> solver(num_x_bins, num_y_bins, width, height);

    solver.density = 1.0;
    solver.kinematic_viscosity = 0.1;
    solver.num_iterations = 1000;
    solver.num_poisson_iterations = 50;
    solver.time_step = 0.001;
    solver.stability_safety_factor = 0.5;

    // establish boundary conditions
    for (int x = 0; x < num_x_bins; x++) {
        solver.setUBoundaryCondition(x, 0, 0.);  // no flow inside of the floor
        solver.setVBoundaryCondition(x, 0, 0.);  // no flow into the floor
        solver.setUBoundaryCondition(x, num_y_bins - 1, 1.);  // water flowing to the right on top
        solver.setVBoundaryCondition(x, num_y_bins - 1, 0.);  // no flow into the top
    }
    for (int y = 0; y < num_y_bins; y++) {
        solver.setUBoundaryCondition(0, y, 0.);  // no flow into of the left wall
        solver.setVBoundaryCondition(0, y, 0.);  // no flow inside the left wall
        solver.setUBoundaryCondition(num_x_bins - 1, y, 0.);  // no floow into the right wall
        solver.setVBoundaryCondition(num_x_bins - 1, y, 0.);  // no flow inside the right wall
    }

    // Solve and retrieve the solution
    solver.solve();
    float* u_values = static_cast<float*>(malloc(sizeof(float) * num_x_bins * num_y_bins));
    float* v_values = static_cast<float*>(malloc(sizeof(float) * num_x_bins * num_y_bins));
    float* p_values = static_cast<float*>(malloc(sizeof(float) * num_x_bins * num_y_bins));
    solver.getUValues(u_values);
    solver.getVValues(v_values);
    solver.getPValues(p_values);

    // write solutions to file
    std::ofstream u_file;
    u_file.open("CppUValues.float.dat", std::ios::binary);
    std::ofstream v_file;
    v_file.open("CppVValues.float.dat", std::ios::binary);
    std::ofstream p_file;
    p_file.open("CppPValues.float.dat", std::ios::binary);
    for (int i = 0; i < num_x_bins * num_y_bins; i++) {
        u_file.write(reinterpret_cast<const char *>(&u_values[i]), sizeof(float));
        u_file.write(reinterpret_cast<const char *>(&v_values[i]), sizeof(float));
        u_file.write(reinterpret_cast<const char *>(&p_values[i]), sizeof(float));
    }

    // clean up
    u_file.close();
    v_file.close();
    p_file.close();
    free(u_values);
    free(v_values);
    free(p_values);


    // run time trials for the solver
    int num_time_trials = 5000;
    float* benchmarks = static_cast<float*>(malloc(sizeof(float) * num_time_trials));
    float compute_time_ms = 0.;
    for (int i = 0; i < num_time_trials; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        solver.solve();
        auto stop = std::chrono::high_resolution_clock::now();

        float duration = float(std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()) * 1e-3;
        benchmarks[i] = duration;
        compute_time_ms += duration;
    }

    // log the results of the time trials to terminal and file
    std::cout << "Time trials complete on average in " << compute_time_ms / static_cast<float>(num_time_trials) << "ms" << std::endl;
    std::ofstream time_file;
    time_file.open("CppBenchmarks.float.dat", std::ios::binary);
    for (int i = 0; i < num_time_trials; i++) {
        time_file.write(reinterpret_cast<const char*>(&benchmarks[i]), sizeof(float));
    }
    time_file.close();
    free(benchmarks);

    return 0;
}
