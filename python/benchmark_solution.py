import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tqdm import tqdm
from functions import *
from navier_stokes import solve_navier_stokes
import time

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

    num_simulations = 5000
    performance_ms = np.zeros(num_simulations)

    # Solve the Navier-Stokes Equation multiple times and time it
    for i in tqdm(range(num_simulations)):
        start = time.perf_counter()
        solve_navier_stokes(num_elements, domain_size, num_iterations, time_step ,
                        kinematic_viscosity, density, top_velocity, num_poisson_iterations,
                        stability_safety_factor)
        stop = time.perf_counter()
        performance_ms[i] = (stop - start) * 1e3

    # plot the data
    plt.figure(figsize=(10, 10), dpi=250)
    plt.hist(performance_ms, bins=int(np.sqrt(num_simulations)))
    plt.xlabel('time (ms)', fontsize=18)
    plt.ylabel('count', fontsize=18)
    average = np.average(performance_ms)
    plt.title('Avg. Completion in {0:.2f} ms'.format(average), fontsize=18)

    plt.savefig('../figures/python_benchmarks.png')

if __name__ == '__main__':
    main()
