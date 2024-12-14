//
// Created by dan on 12/10/24.
//

#include "ParallelNavierStokes.cuh"

template <class T>
ParallelNavierStokes<T>::ParallelNavierStokes(int box_dim_x, int box_dim_y, T domain_size_x, T domain_size_y)
    : NavierStokesSolver<T>(box_dim_x, box_dim_y, domain_size_x, domain_size_y) {
    cudaMalloc((void**)&this->d_cells, sizeof(NavierStokesCell<T>*) * box_dim_x);
    for (int x = 0; x < box_dim_x; x++) {
        cudaMalloc((void*)&(this->cells[x]), sizeof(NavierStokesCell<T>) * box_dim_y);
        cudaMemcpy(this->d_cells[x], this->cells[x], sizeof(NavierStokesCell<T>) * box_dim_y, cudaMemcpyHostToDevice);
    }
}

template <class T>
ParallelNavierStokes<T>::~ParallelNavierStokes() {
    cudaFree(this->d_cells);
}

template <class T>
void ParallelNavierStokes<T>::migrateHostToDevice() {
    for (int x = 0; x < this->box_dimension_x; x++) {
        cudaMemcpy(this->d_cells[x], this->cells[x], sizeof(NavierStokesCell<T>) * this->box_dimension_y, cudaMemcpyHostToDevice);
    }
}


template <class T>
void ParallelNavierStokes<T>::migrateDeviceToHost() {
    for (int x = 0; x < this->box_dimension_x; x++) {
        cudaMemcpy(this->d_cells[x], this->cells[x], sizeof(NavierStokesCell<T>) * this->box_dimension_y, cudaMemcpyHostToDevice);
    }
}


template <class T>
void ParallelNavierStokes<T>::solve() {
    // loop over each time step
    for (int i = 0; i < this->num_iterations; i++) {
        this->unifiedApproximateTimeStep();
        this->unifiedComputeRightHand();

        // take a series of poisson steps to approximate the pressure in each cell
        for (int j = 0; j < this->num_poisson_iterations; j++) {
            // compute the Poisson step, enforce BCs, and enforce the pressure
            this->computePoissonStepApproximation();
            this->enforcePressureBoundaryConditions();
            this->updatePressure();
        }

        // get the pressure central difference, correct the u and v values, and enforce BCs
        this->unifiedVelocityCorrection();
        this->enforceVelocityBoundaryConditions();
    }
}

template <class T>
void ParallelNavierStokes<T>::enforcePressureBoundaryConditions() {
}

template <class T>
void ParallelNavierStokes<T>::updatePressure() {
}

template <class T>
void ParallelNavierStokes<T>::enforceVelocityBoundaryConditions() {
}

template <class T>
void ParallelNavierStokes<T>::unifiedApproximateTimeStep() {
}

template <class T>
void ParallelNavierStokes<T>::unifiedComputeRightHand() {
}

template <class T>
void ParallelNavierStokes<T>::computePoissonStepApproximation() {
}

template <class T>
void ParallelNavierStokes<T>::unifiedVelocityCorrection() {
}
