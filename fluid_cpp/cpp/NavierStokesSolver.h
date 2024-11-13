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

template <typename T>
class NavierStokesSolver {
private:
    int box_dimension_x;
    int box_dimenstion_y;
    T domain_size_x;
    T domain_size_y;
    T element_length_x;
    T element_length_y;

public:
    T time_step;
    T kinematic_viscosity;
    T density;
    int num_poisson_iterations;
    T stability_safety_factor;

    NavierStokesCell<T>* cells;
};



#endif //NAVIERSTOKESSOLVER_H
