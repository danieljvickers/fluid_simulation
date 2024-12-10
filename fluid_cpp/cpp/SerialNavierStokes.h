//
// Created by dan on 12/10/24.
//

#ifndef SERIALNAVIERSTOKES_H
#define SERIALNAVIERSTOKES_H

#include "NavierStokesSolver.h"

template <class T = float>
class SerialNavierStokes : public NavierStokesSolver<T> {
private:
    void computeCentralDifference();
    void computeLaplacian();
    void computeTimeDerivitive();
    void takeTimeStep();
    void computeNextCentralDifference();
    void computeRightHandSide();
    void computePoissonStepApproximation();
    void enforcePressureBoundaryConditions();
    void updatePressure();
    void computePressureCentralDifference();
    void correctVelocityEstimates();
    void enforceVelocityBoundaryConditions();

public:
        SerielNavierStokes(int box_dim_x, int box_dim_y, T domain_size_x, T domain_size_y);
        ~SerielNavierStokes();

    void solve();
};

template class SerielNavierStokes<float>;
template class SerielNavierStokes<double>;

#endif //SERIALNAVIERSTOKES_H
