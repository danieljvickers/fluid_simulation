//
// Created by Daniel J. Vickers on 11/12/24.
//

#ifndef NAVIERSTOKESSOLVER_H
#define NAVIERSTOKESSOLVER_H
#include <cmath>
#include <limits>

// parameters of the cells
template <typename T>
struct NavierStokesCell {
    // the x velcoity (u), y velocity (v), and pressure (p)
    T u = 0.;
    T v = 0.;
    T p = 0.;

    // some values saved in the cell during computation
    T du_dx = 0.;
    T du_dy = 0.;
    T dv_dx = 0.;
    T dv_dy = 0.;
    T u_laplacian = 0.;
    T v_laplacian = 0.;
    T du_dt = 0.;
    T dv_dt = 0.;
    T right_hand_size = 0.;

    // placeholder for the updated values of the sim
    T u_next = 0.;
    T v_next = 0.;
    T p_next = 0.;
    T du_next_dx = 0.;
    T dv_next_dy = 0.;
    T dp_dx = 0.;
    T dp_dy = 0.;

    // boundary conditions (BCs). The set bool tells us if it has been set
    T u_boundary = 0.;
    T v_boundary = 0.;
    T p_boundary = 0.;
    bool u_boundary_set = false;
    bool v_boundary_set = false;
    bool p_boundary_set = false;

};

template <class T = float> class NavierStokesSolver {
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
    void computePoissonStepApproximation(int index_x, int index_y);
    void enforcePressureBoundaryConditions();
    void updatePressure();
    void computePressureCentralDifference(int index_x, int index_y);
    void correctVelocityEstimates(int index_x, int index_y);
    void enforceVelocityBoundaryConditions();

public:
    T time_step = 0.001;
    T kinematic_viscosity = 0.1;
    T density = 1.0;
    int num_poisson_iterations = 50;
    int num_iterations = 1000;
    T stability_safety_factor = 0.5;

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

#endif //NAVIERSTOKESSOLVER_H
