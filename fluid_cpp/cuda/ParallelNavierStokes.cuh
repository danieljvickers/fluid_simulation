//
// Created by dan on 12/10/24.
//

#ifndef PARRALLELNAVIERSTOKES_CUH
#define PARRALLELNAVIERSTOKES_CUH

#include "NavierStokesCell.h"


tamplate <class T>
class ParrallelNavierStokes {
private:
    int box_dimension_x;
    int box_dimension_y;
    T domain_size_x;
    T domain_size_y;
    T element_length_x;
    T element_length_y;

    NavierStokesCell<T>* cells;

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
    T time_step = 0.001;
    T kinematic_viscosity = 0.1;
    T density = 1.0;
    int num_poisson_iterations = 50;
    int num_iterations = 1000;
    T stability_safety_factor = 0.5;
};

template <class T = float> class NavierStokesSolver {



    NavierStokesSolver(int box_dimension_x, int box_dimension_y, T domain_size_x, T domain_size_y);
    ~NavierStokesSolver();
    int setBoxDimenension(int x_dim, int y_dim);
    int setDomainSize(T domain_size_x, T domain_size_y);
    int setUBoundaryCondition(int x_index, int y_index, T BC);
    int setVBoundaryCondition(int x_index, int y_index, T BC);
    int setPBoundaryCondition(int x_index, int y_index, T BC);

    void solve();
    int getCellIndex(int x_index, int y_index);

    int getUValues(T* output);
    int getVValues(T* output);
    int getPValues(T* output);
};

// explicit instantiation allows float and double precision types
template class NavierStokesSolver<float>;
template class NavierStokesSolver<double>;

#endif //PARRALLELNAVIERSTOKES_CUH
