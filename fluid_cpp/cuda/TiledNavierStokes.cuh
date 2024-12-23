//
// Created by dan on 12/23/24.
//

#ifndef TILEDNAVIERSTOKES_CUH
#define TILEDNAVIERSTOKES_CUH

#include "ParallelNavierStokes.cuh"

template <class T>
class TiledNavierStokes : ParallelNavierStokes<T> {
private:
    void tiledApproximateTimeStep();
    void tiledComputeRightHand();
    void tiledComputePoissonStepApproximation();
    void tiledVelocityCorrection();
public:
    TiledNavierStokes(int box_dim_x, int box_dim_y, T domain_size_x, T domain_size_y);
    ~TiledNavierStokes() {};

    void solve();
    void migrateSolve();
};

template class TiledNavierStokes<float>;
template class TiledNavierStokes<double>;

#endif //TILEDNAVIERSTOKES_CUH
