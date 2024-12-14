# Fluid Simulation
Author: Daniel J. Vickers

## About

This repository has been created for me to explore fluid simulation, as an attempt to move towards magnetohydro dynamics simulations. I have opted to start by solving the incompressible Navier-Stokes equations for fluid flow before adding the complexity of electromagnetism. This is also an opportunity for me to demonstrate an ability to write software in Python, C++, and CUDA.

The goal of this project overall is easy-to-read python code that will allow someone to properly learn Navier-Stokes, a C++/CUDA shared object library with python bindings for use in personal projects, and an exploration of benchmarks for performance of the various implementations fo the library.

All of the python code is written to leverage performance from numpy. This is intended to be the unit test for the higher-performance implementations of the simulation and a performance baseline.

The C++ code is meant to be a serial implementation of Navier-Stokes.

The CUDA code is a large-scale parallel implementation of Navier-Stokes. I am leaving several versions of the simulation, and the data presented here is generated using an NVIDIA RTX 2080 Super, and will not likely not yield similar performance on your machine.

## Citations

The Python implementation is based upon an implementation that I found here: https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/lid_driven_cavity_python_simple.py
The results of that solver are captured in a youtube video here: https://www.youtube.com/watch?v=BQLvNLgMTQE
I want it to be clear that this Python code is largely influenced by the contents of the repository I just provided. The C++ and CUDA implementations are my own.
It is easy to find plentiful descriptions of Navier-Stokes online. I will link here the wiki page for the equations: https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations

## Python

The most-basic implementation is a python version that leverages linear algebra in numpy. Because we will be implementing the Cpp and CUDA versions raw (without external libraries), the python implementation is quite optimial.

#### Code Explained

We will be tracking the pressure and velocity (stored as two patrices: u and v) of the fluid in the grid. To start, we instatiate our mesh grid:

```python
# define the x and y values
element_length = domain_size / (num_elements - 1)
x = np.linspace(0., domain_size, num_elements)
y = np.linspace(0., domain_size, num_elements)

# create the vector fields
X, Y = np.meshgrid(x, y)
u_previous = np.zeros_like(X)
v_previous = np.zeros_like(X)
p_previous = np.zeros_like(X)
```

We need the ability to compute the central difference and laplacian on this grid. For that, we define those in functions:

```python
def central_difference_x(field, element_length):
    diff = np.zeros_like(field)
    diff[1:-1, 1:-1] = (field[1:-1, 2:] - field[1:-1, :-2]) / (2. * element_length)
    return diff


def central_difference_y(field, element_length):
    diff = np.zeros_like(field)
    diff[1:-1, 1:-1] = (field[2:, 1:-1] - field[:-2, 1:-1]) / (2. * element_length)
    return diff


def laplace(field, element_length):
    diff = np.zeros_like(field)
    diff[1:-1, 1:-1] = field[1:-1, :-2] + field[:-2, 1:-1] - 4 * field[1:-1, 1:-1] + field[1:-1, 2:] + field[2:, 1:-1]
    diff = diff / (element_length ** 2)
    return diff
```

We can then compute these for our velocities via the Couchy momentum equation:

```python
# get tentative velocity
d_u_previous_d_x = central_difference_x(u_previous, element_length)
d_u_previous_d_y = central_difference_y(u_previous, element_length)
d_v_previous_d_x = central_difference_x(v_previous, element_length)
d_v_previous_d_y = central_difference_y(v_previous, element_length)
u_previous_laplacian = laplace(u_previous, element_length)
v_previous_laplacian = laplace(v_previous, element_length)
```

This gives us some differential information that we can use to plug into the Navier-Stokes equation for velocity and then get a tenative estimate of the new state that we will use in the poison steps:

```python
du_dt = kinematic_viscosity * u_previous_laplacian - u_previous * d_u_previous_d_x - v_previous * d_u_previous_d_y
dv_dt = kinematic_viscosity * v_previous_laplacian - u_previous * d_v_previous_d_x - v_previous * d_v_previous_d_y
u_tentative = u_previous + time_step * du_dt
v_tentative = v_previous + time_step * dv_dt

du_tentative_dx = central_difference_x(u_tentative, element_length)
dv_tentative_dy = central_difference_y(v_tentative, element_length)
```

We then use these velocities to solve the pressure poisson equation. We will perform an iterative number of poison steps to allow the fluid to settle over multiple steps.

```python
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
```

We compute the pressure so that we can use the next equation to update the velocity

```python
dp_next_dx = central_difference_x(p_previous, element_length)
dp_next_dy = central_difference_y(p_previous, element_length)

# correct velocities
u_next = u_tentative - (time_step / density) * dp_next_dx
v_next = v_tentative - (time_step / density) * dp_next_dy
```

Finally we apply our boundary coniditions. Since the python version is only made for an open box, we set the left, right, and bottom velocities to zero. We also set the top velocity to a constant.

Once we have our final velocities, we step forward in time.

```python
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
```

#### Results

The result of the sim is the quiver plot below that captures the fluid velocity and pressure.

![screenshot](figures/python_solution.png)

I then tested the performance of our Navier-Stokes solver. The parameters here are those that match all solutions in this repo:
- 41x41 bins in our grid
- 1000 time steps forward
- 50 poison steps to let the fluid settle

The results of 5000 time trials for the Python version is shown below.

![screenshot](figures/python_benchmarks.png)

In all, this code is capable of performing small simulations and providing reasonable results.

**...But can it go faster?**

## C++

### Serial Implementation
#### Code Explained

For the C++ implementation, we first create a templated class that stores useful data of the simulation, and has useful getters and setters. All of the variables for each cell to make the simulation go fast are given in `fluid_cpp/cpp/NavierStokesCell.h` as:

```c++
template <typename T>
struct NavierStokesCell {
    // the x velcoity (u), y velocity (v), and pressure (p)
    T u = 0.;
    T v = 0.;
    T p = 0.;

    // some values saved in the cell during computation
    T du_dx = 0.;
    T du_dy = 0.;
    T dv_dx = 0.;
    T dv_dy = 0.;
    T u_laplacian = 0.;
    T v_laplacian = 0.;
    T du_dt = 0.;
    T dv_dt = 0.;
    T right_hand_size = 0.;

    // placeholder for the updated values of the sim
    T u_next = 0.;
    T v_next = 0.;
    T p_next = 0.;
    T du_next_dx = 0.;
    T dv_next_dy = 0.;
    T dp_dx = 0.;
    T dp_dy = 0.;

    // boundary conditions (BCs). The set bool tells us if it has been set
    T u_boundary = 0.;
    T v_boundary = 0.;
    T p_boundary = 0.;
    bool u_boundary_set = false;
    bool v_boundary_set = false;
    bool p_boundary_set = false;
};
```

Notice that I am keeping track of some `da_next_db` values because it allows us to perform some computation without iterating through an entire loop again. I then instantiate our deminsion space in the `NavierStokesSolver()` object constructor, which has a private variable of `cells` to hold the parameters. The code for instantiation can be found in `fluid_cpp/cpp/NavierStokesSolver.cpp` file as:

```c++
this->cells = new NavierStokesCell<T>*[box_dim_x];
for (int x = 0; x < box_dim_x; x++) {
    this->cells[x] = new NavierStokesCell<T>[box_dim_y];
}
```

There is a good bit of code here, but the easiest way to show the steps of the computation is to show the `safeSolve()` method, which is a method template with each calculation being done in series. Inside each method is a pair of nested `for` loops, which iterate over each cell in the space. It turns out we can reduce this to only 4 pairs of loops, which is in the `solve()` method, not shown here. Below is the code for `safeSolve()`:

```c++
template <class T>
void SerialNavierStokes<T>::safeSolve() {
    // loop over each time step
    for (int i = 0; i < this->num_iterations; i++) {
        // compute useful derivatives
        this->computeCentralDifference();
        this->computeLaplacian();
        this->computeTimeDerivitive();

        // take a tenative step forward in time
        this->takeTimeStep();
        this->computeNextCentralDifference(); // recompute the central difference
        this->computeRightHandSide();

        // take a series of poisson steps to approximate the pressure in each cell
        for (int j = 0; j < this->num_poisson_iterations; j++) {
            // compute the Poisson step, enforce BCs, and enforce the pressure
            this->computePoissonStepApproximation();
            this->enforcePressureBoundaryConditions();
            this->updatePressure();
        }

        // get the pressure central difference, correct the u and v values, and enforce BCs
        this->computePressureCentralDifference();
        this->correctVelocityEstimates();
        this->enforceVelocityBoundaryConditions();
    }
}
```

When we are ready, we call the code by creating a `SerialNavierStokes` object, setting all of the parameters, establishing boundary conditions, and running the `solve()` method:

```c++
// set up the solver
int num_x_bins = 41;
int num_y_bins = 41;
float width = 1.0;
float height = 1.0;
SerialNavierStokes<float> solver(num_x_bins, num_y_bins, width, height);

// entire specific constants of the simulation
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
```

#### Results

We can view the single-point precision performance on my machine below:

![screenshot](figures/cpp_float_benchmarks.png)

I we see that the C++ version has imporved performance by about a factor of 2.5x. That is a nice performance gain.

**...But can it go even faster?**

### Threaded Implementation

#### Code Explained

#### Results

**But can it go EVEN faster?**

## CUDA

### Initial Implementation

#### Code Explained

This version of the code will function largely by paralellizing individual cell calculations on the interior. Specifically, the test case that we have been demonstrating is 41x41=1681 cells. Rather than assigning a single thread to iterate through the loop, we will assign a single GPU thread to each index in the array of elements, which will perform it's calculations in isolation. We will have some required break points in the code to synchronize the kernel executions and prevent race conditions.

#### Results

So, how did all of that work go?

[screenshot](figures/cuda_float_benchmarks.png)

Now *THAT* is pod racing.

Moving the compute to the GPU allowed us to launch a single thread for each cell in our array. The result is a flat reduction in our compute time. In total, this means launching 1681 total threads for each function that we called.
