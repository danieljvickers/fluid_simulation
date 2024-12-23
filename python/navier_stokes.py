import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from functions import *


'''
In this solver, I am solving the open-top version of Navier-Stokes. This applies the boundary condition
of fluid flowing to the right along the top edge of our box, and the other edges being walls.

This simulation is using a set of common parameters that I found online for reasonable performance, but
I have left these paramters able to be changed.

- num_elements is the number of elements in the X and Y direction. There are a totoal of num_elements^2 bins
- domain_size is the distance in units wide and tall that our box is
- num_iterations is the number of time steps to take
- time_step is the time unit that we are stepping forward
- kinematic_visocsity is a constant of the fluid
- density is also a constant of the fluid
- top_velocity is the speed at which we have fluid flowing across the top of our box
- num_poisson_iterations is the number of equializing steps that we take after each time step
- stability_safety_factor is a constant describing our desired stability of the sim
'''
def solve_navier_stokes(num_elements = 41, domain_size = 1.0, num_iterations = 1000, time_step = 0.001,
                        kinematic_viscosity = 0.1, density = 1., top_velocity = 1., num_poisson_iterations = 50,
                        stability_safety_factor = 0.5):

    # define the x and y values
    element_length = domain_size / (num_elements - 1)
    x = np.linspace(0., domain_size, num_elements)
    y = np.linspace(0., domain_size, num_elements)

    # create the vector fields
    X, Y = np.meshgrid(x, y)
    u_previous = np.zeros_like(X)
    v_previous = np.zeros_like(X)
    p_previous = np.zeros_like(X)

    maximum_possible_time_step_length = (
            0.5 * element_length ** 2 / kinematic_viscosity
    )
    # if time_step > stability_safety_factor * maximum_possible_time_step_length:
    #     raise RuntimeError("Stability is not guarenteed")

    # run the time steps
    for i in range(num_iterations):
        # get tentative velocity
        d_u_previous_d_x = central_difference_x(u_previous, element_length)
        d_u_previous_d_y = central_difference_y(u_previous, element_length)
        d_v_previous_d_x = central_difference_x(v_previous, element_length)
        d_v_previous_d_y = central_difference_y(v_previous, element_length)
        u_previous_laplacian = laplace(u_previous, element_length)
        v_previous_laplacian = laplace(v_previous, element_length)

        du_dt = kinematic_viscosity * u_previous_laplacian - u_previous * d_u_previous_d_x - v_previous * d_u_previous_d_y
        dv_dt = kinematic_viscosity * v_previous_laplacian - u_previous * d_v_previous_d_x - v_previous * d_v_previous_d_y
        u_tentative = u_previous + time_step * du_dt
        v_tentative = v_previous + time_step * dv_dt

        du_tentative_dx = central_difference_x(u_tentative, element_length)
        dv_tentative_dy = central_difference_y(v_tentative, element_length)

        # solve pressure poisson equation
        right_hand_side = (density / time_step) * (du_tentative_dx + dv_tentative_dy)
        p_next = np.zeros_like(p_previous)
        for j in range(num_poisson_iterations):
            p_next[1:-1, 1:-1] = ((right_hand_side[1:-1, 1:-1] * element_length**2) - (
                p_previous[1:-1, :-2] + p_previous[1:-1, 2:] + p_previous[:-2, 1:-1] + p_previous[2:, 1:-1]
            )) * -0.25

            p_next[:, -1] = p_next[:, -2]
            p_next[:, 0] = p_next[:, 1]
            p_next[0, :] = p_next[1, :]
            p_next[-1:, :] = 0.
            p_previous = p_next

        dp_next_dx = central_difference_x(p_previous, element_length)
        dp_next_dy = central_difference_y(p_previous, element_length)

        # correct velocities
        u_next = u_tentative - (time_step / density) * dp_next_dx
        v_next = v_tentative - (time_step / density) * dp_next_dy

        # apply the boundary conditions
        u_next[0, :] = 0.  # bottom
        u_next[-1, :] = top_velocity  # top
        u_next[:, 0] = 0.  # left
        u_next[:, -1] = 0.  # right
        v_next[0, :] = 0.
        v_next[-1, :] = 0.
        v_next[:, 0] = 0.
        v_next[:, -1] = 0.

        # step forward in time
        u_previous = u_next
        v_previous = v_next
        p_previous = p_next
    
    return(X, Y, u_previous, v_previous, p_previous)