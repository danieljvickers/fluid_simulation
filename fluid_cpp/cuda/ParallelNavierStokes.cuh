//
// Created by dan on 12/10/24.
//

#ifndef PARALLELNAVIERSTOKES_CUH
#define PARALLELNAVIERSTOKES_CUH

#include "../cpp/NavierStokesCell.h"
#include "../cpp/NavierStokesSolver.h"

#define KERNEL_2D_WIDTH 16
#define KERNEL_2D_HEIGHT 16

template <class T>
class ParallelNavierStokes : public NavierStokesSolver<T> {
private:
    NavierStokesCell<T>* d_cells;

    void enforcePressureBoundaryConditions();
    void updatePressure();
    void enforceVelocityBoundaryConditions();

    void unifiedApproximateTimeStep();
    void unifiedComputeRightHand();
    void computePoissonStepApproximation();
    void unifiedVelocityCorrection();

public:
    ParallelNavierStokes(int box_dim_x, int box_dim_y, T domain_size_x, T domain_size_y);
    ~ParallelNavierStokes();

    void migrateHostToDevice();
    void migrateDeviceToHost();
    void solve();
};

// explicit instantiation allows float and double precision types
template class ParallelNavierStokes<float>;
template class ParallelNavierStokes<double>;

#endif //PARALLELNAVIERSTOKES_CUH
