//
// Created by Daniel J. Vickers on 11/12/24.
//

#ifndef NAVIERSTOKESSOLVER_H
#define NAVIERSTOKESSOLVER_H
#include "NavierStokesCell.h"

template <class T = float>
class NavierStokesSolver {
protected:
    int box_dimension_x;
    int box_dimension_y;
    T domain_size_x;
    T domain_size_y;
    T element_length_x;
    T element_length_y;

    NavierStokesCell<T>** cells;

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
    int setUBoundaryCondition(int x, int y, T BC);
    int setVBoundaryCondition(int x, int y, T BC);
    int setPBoundaryCondition(int x, int y, T BC);

    int getUValues(T* output);
    int getVValues(T* output);
    int getPValues(T* output);
};

// explicit instantiation allows float and double precision types
template class NavierStokesSolver<float>;
template class NavierStokesSolver<double>;

#endif //NAVIERSTOKESSOLVER_H
