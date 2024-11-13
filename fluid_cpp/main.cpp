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
    std::cout << "Time trials complete on average in " << compute_time_ms / num_time_trials << "ms" << std::endl;
    std::ofstream file;
    file.open("CppBenchmarks.float.dat", std::ios::binary);
    for (int i = 0; i < num_time_trials; i++) {
        file.write(reinterpret_cast<const char*>(&benchmarks[i]), sizeof(float));
    }

    free(benchmarks);

    return 0;
}
