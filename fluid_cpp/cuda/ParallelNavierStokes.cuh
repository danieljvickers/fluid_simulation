//
// Created by dan on 12/10/24.
//

#ifndef PARALLELNAVIERSTOKES_CUH
#define PARALLELNAVIERSTOKES_CUH

#include "../cpp/NavierStokesCell.h"
#include "../cpp/NavierStokesSolver.h"

#define KERNEL_2D_WIDTH 4
#define KERNEL_2D_HEIGHT 4
#define GRID_2D_WIDTH 4
#define GRID_2D_HEIGHT 4

template <class T>
class ParallelNavierStokes : public NavierStokesSolver<T> {
private:
    NavierStokesCell<T>* d_cells;
    T* d_u;
    T* d_v;
    T* d_u_temp;
    T* d_v_temp;
    T* d_p;
    dim3 block_size;
    dim3 grid_size;

    void enforcePressureBoundaryConditions();
    void updatePressure();
    void enforceVelocityBoundaryConditions();

    void unifiedApproximateTimeStep();
    void unifiedComputeRightHand();
    void computePoissonStepApproximation();
    void unifiedVelocityCorrection();

    void createKernelDims();

public:
    ParallelNavierStokes(int box_dim_x, int box_dim_y, T domain_size_x, T domain_size_y);
    ~ParallelNavierStokes();

    void migrateHostToDevice();
    void migrateDeviceToHost();
    void solve();
    void migrateSolve();
};

// explicit instantiation allows float and double precision types
template class ParallelNavierStokes<float>;
template class ParallelNavierStokes<double>;

#endif //PARALLELNAVIERSTOKES_CUH
