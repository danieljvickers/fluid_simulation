#include <iostream>
#include "cpp/NavierStokesSolver.h"

int main() {
    int num_x_bins = 41;
    int num_y_bins = 41;
    float width = 1.0;
    float height = 1.0;

    NavierStokesSolver solver(num_x_bins, num_y_bins, width, height);

    solver.density = 1.0;
    solver.kinematic_viscosity = 0.1;
    solver.num_iterations = 1000;
    solver.num_poisson_iterations = 50;
    solver.time_step = 0.001;
    solver.stability_safety_factor = 0.5;

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
