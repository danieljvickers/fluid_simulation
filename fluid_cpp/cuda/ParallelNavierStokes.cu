//
// Created by dan on 12/10/24.
//

#include "ParallelNavierStokes.cuh"

template <class T>
ParallelNavierStokes<T>::ParallelNavierStokes(int box_dim_x, int box_dim_y, T domain_size_x, T domain_size_y)
    : NavierStokesSolver<T>(box_dim_x, box_dim_y, domain_size_x, domain_size_y) {
    cudaMalloc((void**)&this->d_cells, sizeof(NavierStokesCell<T>) * box_dim_x * box_dim_y);
    cudaMemcpy(this->d_cells, this->cells, sizeof(NavierStokesCell<T>) * box_dim_x * box_dim_y, cudaMemcpyHostToDevice);
}

template <class T>
ParallelNavierStokes<T>::~ParallelNavierStokes() {
    cudaFree(this->d_cells);
}
