//
// Created by dan on 12/18/24.
//

#ifndef THREADEDNAVIERSTOKES_H
#define THREADEDNAVIERSTOKES_H

#include "NavierStokesSolver.h"
#include <thread>
#include <shared_mutex>
#include <barrier>

#define THREADED_GRID_SIZE 8

template <class T = float>
class ThreadedNavierStokes : public NavierStokesSolver<T> {
private:
    void computePoissonStepApproximation(int thread_index);
    void enforcePressureBoundaryConditions(int thread_index);
    void updatePressure(int thread_index);
    void enforceVelocityBoundaryConditions(int thread_index);

    void unifiedApproximateTimeStep(int thread_index);
    void unifiedComputeRightHand(int thread_index);
    void unifiedVelocityCorrection(int thread_index);

    void syncThreads();
    void solveThread(int index);

    std::thread* worker_threads;
    std::shared_mutex sync_mutex;
    int lock_count = 0;

public:
    ThreadedNavierStokes(int box_dim_x, int box_dim_y, T domain_size_x, T domain_size_y);
    ~ThreadedNavierStokes();

    void solve();
};

template class ThreadedNavierStokes<float>;
template class ThreadedNavierStokes<double>;


#endif //THREADEDNAVIERSTOKES_H
