//
// Created by dan on 12/23/24.
//

#include "TiledNavierStokes.cuh"
#include "../cpp/NavierStokesCell.h"

/*
    Kernels
*/
template <typename T>
__global__ void tiled_timestep_kernel(NavierStokesCell<T>* cells, int width, int height, T element_length_x, T element_length_y, T kinematic_viscosity, T time_step) {
    // get our location in the grid
    const int column = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

    __shared__ NavierStokesCell<T> s_cells[KERNEL_2D_WIDTH * KERNEL_2D_HEIGHT];
    int s_index = threadIdx.x * KERNEL_2D_HEIGHT + threadIdx.y;

    for (int c = column; c < width - 1; c += gridDim.x * blockDim.x) {
        for (int r = row; r < height - 1; r += gridDim.y * blockDim.y) {
            int index = c * height + r;
            s_cells[(threadIdx.x) * KERNEL_2D_HEIGHT + threadIdx.y] = cells[index];
            __syncthreads();

            NavierStokesCell<T> up;
            NavierStokesCell<T> down;
            NavierStokesCell<T> right;
            NavierStokesCell<T> left;
            if (threadIdx.y == 0) down = cells[index - 1];
            else down = s_cells[s_index - 1];
            if (threadIdx.y == KERNEL_2D_HEIGHT - 1) up = cells[index + 1];
            else up = s_cells[s_index + 1];
            if (threadIdx.x == 0) left = cells[index - height];
            else left = s_cells[s_index - 1];
            if (threadIdx.x == KERNEL_2D_WIDTH - 1) right = cells[index + height];
            else right = s_cells[s_index + KERNEL_2D_HEIGHT];
            __syncthreads();

            // compute the central differences
            T du_dx = (right.u - left.u) / 2. / element_length_x;
            T dv_dx = (right.v - left.v) / 2. / element_length_x;
            T du_dy = (up.u - down.u) / 2. / element_length_y;
            T dv_dy = (up.v - down.v) / 2. / element_length_y;

            // compute the laplacian
            T u_laplacian = left.u + right.u + up.u + down.u;
            u_laplacian = (u_laplacian - 4. * s_cells[s_index].u) / element_length_x / element_length_y;
            T v_laplacian = left.v + right.v + up.v + down.v;
            v_laplacian = (v_laplacian - 4. * s_cells[s_index].v) / element_length_x / element_length_y;

            // get the time derivitives
            T du_dt = kinematic_viscosity * u_laplacian - s_cells[s_index].u * du_dx - s_cells[s_index].v * du_dy;
            T dv_dt = kinematic_viscosity * v_laplacian - s_cells[s_index].u * dv_dx - s_cells[s_index].v * dv_dy;

            // step forward in time
            cells[index].u_next = s_cells[s_index].u + time_step * du_dt;
            cells[index].v_next = s_cells[s_index].v + time_step * dv_dt;
        }
    }
}

template <typename T>
__global__ void tiled_compute_righthand_kernel(NavierStokesCell<T>* cells, int width, int height, T element_length_x, T element_length_y, T density, T time_step) {
    // get our location in the grid
    const int column = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

    for (int c = column; c < width - 1; c += gridDim.x * blockDim.x) {
        for (int r = row; r < height - 1; r += gridDim.y * blockDim.y) {
            int index = c * height + r;
            int up = index + 1;
            int down = index - 1;
            int left = index - height;
            int right = index + height;

            // compute the central differences
            cells[index].du_next_dx = (cells[right].u_next - cells[left].u_next) / 2. / element_length_x;
            cells[index].dv_next_dy = (cells[up].v_next - cells[down].v_next) / 2. / element_length_y;
            cells[index].right_hand_size = (density / time_step) * (cells[index].du_next_dx + cells[index].dv_next_dy);
        }
    }
}

template <typename T>
__global__ void tiled_poisson_step_kernel(NavierStokesCell<T>* cells, int width, int height, T element_length_x, T element_length_y) {
    // get our location in the grid
    const int column = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

    __shared__ NavierStokesCell<T> s_cells[KERNEL_2D_WIDTH * KERNEL_2D_HEIGHT];
    int s_index = threadIdx.x * KERNEL_2D_HEIGHT + threadIdx.y;

    for (int c = column; c < width - 1; c += gridDim.x * blockDim.x) {
        for (int r = row; r < height - 1; r += gridDim.y * blockDim.y) {
            int index = c * height + r;
            s_cells[(threadIdx.x) * KERNEL_2D_HEIGHT + threadIdx.y] = cells[index];
            __syncthreads();

            NavierStokesCell<T> up;
            NavierStokesCell<T> down;
            NavierStokesCell<T> right;
            NavierStokesCell<T> left;
            if (threadIdx.y == 0) down = cells[index - 1];
            else down = s_cells[s_index - 1];
            if (threadIdx.y == KERNEL_2D_HEIGHT - 1) up = cells[index + 1];
            else up = s_cells[s_index + 1];
            if (threadIdx.x == 0) left = cells[index - height];
            else left = s_cells[s_index - 1];
            if (threadIdx.x == KERNEL_2D_WIDTH - 1) right = cells[index + height];
            else right = s_cells[s_index + KERNEL_2D_HEIGHT];
            __syncthreads();

            T p_next = s_cells[s_index].right_hand_size * element_length_x * element_length_y;
            p_next -= left.p + right.p + up.p + down.p;
            cells[index].p_next = p_next * -0.25;
        }
    }
}

template <typename T>
__global__ void tiled_velocity_correction_kerenl(NavierStokesCell<T>* cells, int width, int height, T element_length_x, T element_length_y, T density, T time_step) {
    // get our location in the grid
    const int column = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

    for (int c = column; c < width - 1; c += gridDim.x * blockDim.x) {
        for (int r = row; r < height - 1; r += gridDim.y * blockDim.y) {
            int index = c * height + r;
            int up = index + 1;
            int down = index - 1;
            int left = index - height;
            int right = index + height;

            // compute the central differences
            cells[index].dp_dx = (cells[right].p - cells[left].p) / 2. / element_length_x;
            cells[index].dp_dy = (cells[up].p - cells[down].p) / 2. / element_length_y;

            // compute final velocity
            cells[index].u = cells[index].u_next - (time_step / density) * cells[index].dp_dx;
            cells[index].v = cells[index].v_next - (time_step / density) * cells[index].dp_dy;
        }
    }
}

/*
    Class Methods
*/

template <class T>
TiledNavierStokes<T>::TiledNavierStokes(int box_dim_x, int box_dim_y, T domain_size_x, T domain_size_y)
    : ParallelNavierStokes<T>(box_dim_x, box_dim_y, domain_size_x, domain_size_y) {
}

template <class T>
void TiledNavierStokes<T>::solve() {
    // loop over each time step
    for (int i = 0; i < this->num_iterations; i++) {
        this->tiledApproximateTimeStep();
        this->tiledComputeRightHand();

        // take a series of poisson steps to approximate the pressure in each cell
        for (int j = 0; j < this->num_poisson_iterations; j++) {
            // compute the Poisson step, enforce BCs, and enforce the pressure
            this->tiledComputePoissonStepApproximation();
            this->enforcePressureBoundaryConditions();
            this->updatePressure();
        }

        // get the pressure central difference, correct the u and v values, and enforce BCs
        this->tiledVelocityCorrection();
        this->enforceVelocityBoundaryConditions();
    }
}

template <class T>
void TiledNavierStokes<T>::migrateSolve() {
    this->migrateHostToDevice();
    this->solve();
    this->migrateDeviceToHost();
}


template <class T>
void TiledNavierStokes<T>::tiledApproximateTimeStep() {
    tiled_timestep_kernel<T><<<this->grid_size, this->block_size>>>(this->d_cells, this->box_dimension_x, this->box_dimension_y,
        this->element_length_x,this->element_length_y, this->kinematic_viscosity, this->time_step);
}

template <class T>
void TiledNavierStokes<T>::tiledComputeRightHand() {
    tiled_compute_righthand_kernel<T><<<this->grid_size, this->block_size>>>(this->d_cells, this->box_dimension_x, this->box_dimension_y,
        this->element_length_x, this->element_length_y, this->density, this->time_step);
}

template <class T>
void TiledNavierStokes<T>::tiledComputePoissonStepApproximation() {
    tiled_poisson_step_kernel<T><<<this->grid_size, this->block_size>>>(this->d_cells, this->box_dimension_x, this->box_dimension_y,
        this->element_length_x, this->element_length_y);
}

template <class T>
void TiledNavierStokes<T>::tiledVelocityCorrection() {
    tiled_velocity_correction_kerenl<T><<<this->grid_size, this->block_size>>>(this->d_cells, this->box_dimension_x, this->box_dimension_y,
        this->element_length_x, this->element_length_y, this->density, this->time_step);
}