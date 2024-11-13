//
// Created by Daniel J. Vickers on 11/12/24.
//

#ifndef NAVIERSTOKESSOLVER_H
#define NAVIERSTOKESSOLVER_H
#include <cmath>

// parameters of the cells
template <typename T>
struct NavierStokesCell {
    // the x velcoity (u), y velocity (v), and pressure (p)
    T u;
    T v;
    T p;

    // some values saved in the cell during computation
    T d_u_d_x;
    T d_u_d_y;
    T d_v_d_x;
    T d_v_d_y;
    T u_laplacian;
    T v_laplacian;
    T du_dt;
    T dv_dt;
    T right_hand_size;

    // placeholder for the updated values of the sim
    T u_next;
    T v_next;
    T p_next;

    // boundary conditions (BCs). NAN means no BC
    T u_boundary = NAN;
    T v_boundary = NAN;
    T p_boundary = NAN;
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

public:
    T time_step = 0.001;
    T kinematic_viscosity = 0.1;
    T density = 1.0;
    int num_poisson_iterations = 50;
    int num_iterations = 1000;
    T stability_safety_factor = 0.5;

    NavierStokesSolver(int box_dimension_x, int box_dimension_y, T domain_size_x, T domain_size_y);
    int setBoxDimenension(int x_dim, int y_dim);
    int setDomainSize(T domain_size_x, T domain_size_y);
};

// explicit instantiation allows float and double precision types
template class NavierStokesSolver<float>;
template class NavierStokesSolver<double>;

#endif //NAVIERSTOKESSOLVER_H
