//
// Created by Daniel J. Vickers on 11/12/24.
//

#include "NavierStokesSolver.h"

template <class T>
NavierStokesSolver<T>::NavierStokesSolver(int box_dim_x, int box_dim_y, T domain_size_x, T domain_size_y) {
    box_dimension_x = box_dim_x;
    box_dimension_y = box_dim_y;
    this->cells = static_cast<NavierStokesCell<T>*>(malloc(sizeof(NavierStokesCell<T>) * box_dimension_x * box_dimension_y));

    this->setDomainSize(domain_size_x, domain_size_x);
}

template <class T>
int NavierStokesSolver<T>::setBoxDimenension(int box_dimension_x, int box_dimension_y) {
    if (box_dimension_x == this->box_dimension_x && box_dimension_y == this->box_dimension_y) {
        return -1;
    }

    free(this->cells);
    this->box_dimension_x = box_dimension_x;
    this->box_dimension_y = box_dimension_y;
    this->cells = static_cast<NavierStokesCell<T>*>(malloc(sizeof(NavierStokesCell<T>) * box_dimension_x * box_dimension_y));
    this->setDomainSize(this->domain_size_x, this->domain_size_y);
    return 0;
}

template <class T>
int NavierStokesSolver<T>::setDomainSize(T domain_size_x, T domain_size_y) {
    this->domain_size_x = domain_size_x;
    this->domain_size_y = domain_size_y;

    this->element_length_x = domain_size_x / (this->box_dimension_x - 1);
    this->element_length_y = domain_size_y / (this->box_dimension_y - 1);

    return 0;
}
