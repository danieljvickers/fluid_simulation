import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# from tqdm import tqdm
from functions import *
from navier_stokes import solve_navier_stokes

# matplotlib.use('TkAgg')


def main():
    # constants of the sim
    num_elements = 41
    domain_size = 1.0
    num_iterations = 1000
    time_step = 0.001
    kinematic_viscosity = 0.1
    density = 1.
    top_velocity = 1.
    num_poisson_iterations = 50
    stability_safety_factor = 0.5

    # Solve the Navier-Stokes Equation once
    (X, Y, u, v, p) = solve_navier_stokes(num_elements, domain_size, num_iterations, time_step ,
                        kinematic_viscosity, density, top_velocity, num_poisson_iterations,
                        stability_safety_factor)

    # plot the data
    plt.figure(figsize=(10, 10), dpi=250)
    plt.contourf(X, Y, p, cmap='coolwarm')
    cbar = plt.colorbar()
    cbar.set_label('Pressure', rotation=270)

    plt.quiver(X[::2], Y[::2], u[::2], v[::2], color='black')
    plt.xlim([0, domain_size])
    plt.ylim([0, domain_size])
    plt.savefig('../figures/python_solution.png')

if __name__ == '__main__':
    main()
