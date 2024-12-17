//
// Created by dan on 12/10/24.
//

#include "ParallelNavierStokes.cuh"

#include <cuda_runtime_api.h>
#include <iostream>

/*
 CUDA Kernels to run on GPU
*/

template <typename T> // TODO :: Update Later
__global__ void enforce_pressure_BC_kernel(NavierStokesCell<T>* cells, int width, int height) {
    // get our location in the grid
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    for (int c = column; c < width; c += gridDim.x * blockDim.x) {
        for (int r = row; r < height; r += gridDim.y * blockDim.y) {
            int index = c * height + r;

            // return if you have a BC
            if (cells[index].p_boundary_set) {
                cells[index].p_next = cells[index].p_boundary;  // enforce the BC if it has been set
                continue;
            }

            // checks if you are on the edge, else do nothing
            if (r == 0) {
                cells[index].p_next = cells[index + 1].p_next; // equal to cell above
            } else if (r == width - 1) {
                cells[index].p_next = cells[index - 1].p_next; // equal to cell below
            } else if (c == 0) {
                cells[index].p_next = cells[index + height].p_next; // equal to cell to right
            } else if (c == height - 1) {
                cells[index].p_next = cells[index - height].p_next;  // equal to cell to left
            }
        }
    }
}

template <typename T>
__global__ void update_pressure_kernel(NavierStokesCell<T>* cells, int width, int height) {
    // get our location in the grid
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    for (int c = column; c < width; c += gridDim.x * blockDim.x) {
        for (int r = row; r < height; r += gridDim.y * blockDim.y) {
            int index = c * height + r;
            cells[index].p = cells[index].p_next;
        }
    }
}

template <typename T> // TODO :: Update Later
__global__ void enforce_velocity_BC_kernel(NavierStokesCell<T>* cells, int width, int height) {
    // get our location in the grid
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    // iterate over u velocity values
    for (int c = column; c < width; c += gridDim.x * blockDim.x) {
        for (int r = row; r < height; r += gridDim.y * blockDim.y) {
            int index = c * height + r;

            // return if you have a BC
            if (cells[index].u_boundary_set) {
                cells[index].u = cells[index].u_boundary;  // enforce the BC if it has been set
                return;
            }

            // checks if you are on the edge, else do nothing
            if (r == 0) {
                cells[index].u = cells[index + 1].u; // equal to cell above
            } else if (r == width - 1) {
                cells[index].u = cells[index - 1].u; // equal to cell below
            } else if (c == 0) {
                cells[index].u = cells[index + height].u; // equal to cell to right
            } else if (c == height - 1) {
                cells[index].u = cells[index - height].u;  // equal to cell to left
            }
        }
    }

    // iterate over v velocity values
    for (int c = column; c < width; c += gridDim.x * blockDim.x) {
        for (int r = row; r < height; r += gridDim.y * blockDim.y) {
            int index = c * height + r;

            // return if you have a BC
            if (cells[index].v_boundary_set) {
                cells[index].v = cells[index].v_boundary;  // enforce the BC if it has been set
                return;
            }

            // checks if you are on the edge, else do nothing
            if (r == 0) {
                cells[index].v = cells[index + 1].v; // equal to cell above
            } else if (r == width - 1) {
                cells[index].v = cells[index - 1].v; // equal to cell below
            } else if (c == 0) {
                cells[index].v = cells[index + height].v; // equal to cell to right
            } else if (c == height - 1) {
                cells[index].v = cells[index - height].v;  // equal to cell to left
            }
        }
    }
}

template <typename T>
__global__ void unified_timestep_kernel(NavierStokesCell<T>* cells, int width, int height, T element_length_x, T element_length_y, T kinematic_viscosity, T time_step) {
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
            T du_dx = (cells[right].u - cells[left].u) / 2. / element_length_x;
            T dv_dx = (cells[right].v - cells[left].v) / 2. / element_length_x;
            T du_dy = (cells[up].u - cells[down].u) / 2. / element_length_y;
            T dv_dy = (cells[up].v - cells[down].v) / 2. / element_length_y;

            // compute the laplacian
            T u_laplacian = cells[left].u + cells[right].u + cells[up].u + cells[down].u;
            u_laplacian = (u_laplacian - 4. * cells[index].u) / element_length_x / element_length_y;
            T v_laplacian = cells[left].v + cells[right].v + cells[up].v + cells[down].v;
            v_laplacian = (v_laplacian - 4. * cells[index].v) / element_length_x / element_length_y;

            // get the time derivitives
            T du_dt = kinematic_viscosity * u_laplacian - cells[index].u * du_dx - cells[index].v * du_dy;
            T dv_dt = kinematic_viscosity * v_laplacian - cells[index].u * dv_dx - cells[index].v * dv_dy;

            // step forward in time
            cells[index].u_next = cells[index].u + time_step * du_dt;
            cells[index].v_next = cells[index].v + time_step * dv_dt;
        }
    }
}

template <typename T>
__global__ void vector_time_step(T* u, T* v, T* u_temp, T* v_temp, int width, int height, T element_length_x, T element_length_y, T kinematic_viscosity, T time_step) {
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

            T u_current = u[index];
            T u_up = u[up];
            T u_down = u[down];
            T u_left = u[left];
            T u_right = u[right];
            T v_current = v[index];
            T v_up = v[up];
            T v_down = v[down];
            T v_left = v[left];
            T v_right = v[right];

            // compute the central differences  // 200 ms
            T du_dx = (u_right - u_left) / 2. / element_length_x;
            T dv_dx = (v_right - v_left) / 2. / element_length_x;
            T du_dy = (u_up - u_down) / 2. / element_length_y;
            T dv_dy = (v_up - v_down) / 2. / element_length_y;

            // compute the laplacian  // 380 ms
            T u_laplacian = u_left + u_right + u_up + u_down;
            u_laplacian = (u_laplacian - 4. * u_current) / element_length_x / element_length_y;
            T v_laplacian = v_left + v_right + v_up + v_down;
            v_laplacian = (v_laplacian - 4. * v_current) / element_length_x / element_length_y;

            // get the time derivitives  // 755 ms
            T du_dt = kinematic_viscosity * u_laplacian - u_current * du_dx - v_current * du_dy;
            T dv_dt = kinematic_viscosity * v_laplacian - u_current * dv_dx - v_current * dv_dy;

            // step forward in time  // 755 ms
            u_temp[index] = u_current + time_step * du_dt;
            v_temp[index] = v_current + time_step * dv_dt;
        }
    }
}

template <typename T>
__global__ void compute_righthand_kernel(NavierStokesCell<T>* cells, int width, int height, T element_length_x, T element_length_y, T density, T time_step) {
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
__global__ void poisson_step_kernel(NavierStokesCell<T>* cells, int width, int height, T element_length_x, T element_length_y) {
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

            // compute the Poisson step
            T p_next = cells[index].right_hand_size * element_length_x * element_length_y;
            p_next -= cells[left].p + cells[right].p + cells[up].p + cells[down].p;
            cells[index].p_next = p_next * -0.25;
        }
    }
}

template <typename T>
__global__ void velocity_correction_kerenl(NavierStokesCell<T>* cells, int width, int height, T element_length_x, T element_length_y, T density, T time_step) {
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
 Class Methods to call kernel code
*/

template <class T>
ParallelNavierStokes<T>::ParallelNavierStokes(int box_dim_x, int box_dim_y, T domain_size_x, T domain_size_y)
    : NavierStokesSolver<T>(box_dim_x, box_dim_y, domain_size_x, domain_size_y) {
    cudaMalloc((void**)&this->d_cells, sizeof(NavierStokesCell<T>) * box_dim_x * box_dim_y);
    cudaMalloc((void**)&this->d_u, sizeof(T) * box_dim_x * box_dim_y);
    cudaMalloc((void**)&this->d_v, sizeof(T) * box_dim_x * box_dim_y);
    cudaMalloc((void**)&this->d_u_temp, sizeof(T) * box_dim_x * box_dim_y);
    cudaMalloc((void**)&this->d_v_temp, sizeof(T) * box_dim_x * box_dim_y);
    cudaMalloc((void**)&this->d_p, sizeof(T) * box_dim_x * box_dim_y);

    for (int x = 0; x < this->box_dimension_x; x++) {
        cudaMemcpy(&(this->d_cells[x * this->box_dimension_y]), this->cells[x], sizeof(NavierStokesCell<T>) * this->box_dimension_y, cudaMemcpyHostToDevice);
    }
    this->createKernelDims();
}

template <class T>
ParallelNavierStokes<T>::~ParallelNavierStokes() {
    cudaFree(this->d_cells);
}

template <class T>
void ParallelNavierStokes<T>::migrateHostToDevice() {
    for (int x = 0; x < this->box_dimension_x; x++) {
        cudaMemcpy(&(this->d_cells[x * this->box_dimension_y]), this->cells[x], sizeof(NavierStokesCell<T>) * this->box_dimension_y, cudaMemcpyHostToDevice);
    }
}


template <class T>
void ParallelNavierStokes<T>::migrateDeviceToHost() {
    for (int x = 0; x < this->box_dimension_x; x++) {
        cudaMemcpy(this->cells[x], &(this->d_cells[x * this->box_dimension_y]), sizeof(NavierStokesCell<T>) * this->box_dimension_y, cudaMemcpyDeviceToHost);
    }
}

template <class T>
void ParallelNavierStokes<T>::createKernelDims() {
    dim3 block_size(KERNEL_2D_WIDTH, KERNEL_2D_HEIGHT);  // compute the size of each block
    int bx = (this->box_dimension_x + block_size.x - 1) / block_size.x;  // x size in blocks of the grid
    int by = (this->box_dimension_y + block_size.y - 1) / block_size.y;  // y size in blocks of the grid
    dim3 grid_size = dim3(GRID_2D_WIDTH, GRID_2D_HEIGHT);
    // dim3 grid_size = dim3(bx, by);

    this->block_size = block_size;
    this->grid_size = grid_size;
}


template <class T>
void ParallelNavierStokes<T>::solve() {
    // loop over each time step
    for (int i = 0; i < this->num_iterations; i++) {
        this->unifiedApproximateTimeStep();  // 771
        this->unifiedComputeRightHand();  // 197

        // take a series of poisson steps to approximate the pressure in each cell
        for (int j = 0; j < this->num_poisson_iterations; j++) {
            // compute the Poisson step, enforce BCs, and enforce the pressure
            this->computePoissonStepApproximation();  // 24.29 :: 1023.3
            this->enforcePressureBoundaryConditions();  // 1.457 :: 66.5
            this->updatePressure();  // 6.604 :: 265.881
        }

        // get the pressure central difference, correct the u and v values, and enforce BCs
        this->unifiedVelocityCorrection();  // 217.622
        this->enforceVelocityBoundaryConditions();  // 1.46
    }
}

template <class T>
void ParallelNavierStokes<T>::migrateSolve() {
    this->migrateHostToDevice();
    this->solve();
    this->migrateDeviceToHost();
}

template <class T>
void ParallelNavierStokes<T>::enforcePressureBoundaryConditions() {
    enforce_pressure_BC_kernel<T><<<this->grid_size, this->block_size>>>(this->d_cells, this->box_dimension_x, this->box_dimension_y);
}

template <class T>
void ParallelNavierStokes<T>::updatePressure() {
    update_pressure_kernel<T><<<this->grid_size, this->block_size>>>(this->d_cells, this->box_dimension_x, this->box_dimension_y);
}

template <class T>
void ParallelNavierStokes<T>::enforceVelocityBoundaryConditions() {
    enforce_velocity_BC_kernel<T><<<this->grid_size, this->block_size>>>(this->d_cells, this->box_dimension_x, this->box_dimension_y);
}

template <class T>
void ParallelNavierStokes<T>::unifiedApproximateTimeStep() {
    // vector_time_step<T><<<this->grid_size, this->block_size>>>(this->d_u, this->d_v, this->d_u_temp, this->d_v_temp, this->box_dimension_x, this->box_dimension_y,
    //     this->element_length_x,this->element_length_y, this->kinematic_viscosity, this->time_step);

    unified_timestep_kernel<T><<<this->grid_size, this->block_size>>>(this->d_cells, this->box_dimension_x, this->box_dimension_y,
        this->element_length_x,this->element_length_y, this->kinematic_viscosity, this->time_step);
}

template <class T>
void ParallelNavierStokes<T>::unifiedComputeRightHand() {
    compute_righthand_kernel<T><<<this->grid_size, this->block_size>>>(this->d_cells, this->box_dimension_x, this->box_dimension_y,
        this->element_length_x, this->element_length_y, this->density, this->time_step);
}

template <class T>
void ParallelNavierStokes<T>::computePoissonStepApproximation() {
    poisson_step_kernel<T><<<this->grid_size, this->block_size>>>(this->d_cells, this->box_dimension_x, this->box_dimension_y,
        this->element_length_x, this->element_length_y);
}

template <class T>
void ParallelNavierStokes<T>::unifiedVelocityCorrection() {
    velocity_correction_kerenl<T><<<this->grid_size, this->block_size>>>(this->d_cells, this->box_dimension_x, this->box_dimension_y,
        this->element_length_x, this->element_length_y, this->density, this->time_step);
}
