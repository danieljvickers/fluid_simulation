//
// Created by Daniel J. Vickers on 11/12/24.
//

#include "NavierStokesSolver.h"

#include <ios>
#include <iostream>

template <class T>
NavierStokesSolver<T>::NavierStokesSolver(int box_dim_x, int box_dim_y, T domain_size_x, T domain_size_y) {
    box_dimension_x = box_dim_x;
    box_dimension_y = box_dim_y;
    this->cells = static_cast<NavierStokesCell<T>**>(malloc(sizeof(NavierStokesCell<T>*) * box_dim_x));
    for (int x = 0; x < box_dim_x; x++) {
        this->cells[x] = static_cast<NavierStokesCell<T>>(malloc(sizeof(NavierStokesCell<T>) * box_dim_y));
    }

    this->setDomainSize(domain_size_x, domain_size_x);
}

template <class T>
NavierStokesSolver<T>::~NavierStokesSolver() {
    for (int x = 0; x < this->box_dimension_x; x++) {
        free(cells[x]);
    }
    free(this->cells);
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

template <class T>
int NavierStokesSolver<T>::setUBoundaryCondition(int const x_index, int const y_index, T const BC) {
    int index = getCellIndex(x_index, y_index);
    this->cells[index].u_boundary = BC;
    this->cells[index].u = BC;
    this->cells[index].u_boundary_set = true;
    return 0;
}

template <class T>
int NavierStokesSolver<T>::setVBoundaryCondition(int const x_index, int const y_index, T const BC) {
    int index = getCellIndex(x_index, y_index);
    this->cells[index].v_boundary = BC;
    this->cells[index].v = BC;
    this->cells[index].v_boundary_set = true;
    return 0;
}

template <class T>
int NavierStokesSolver<T>::setPBoundaryCondition(int const x_index, int const y_index, T const BC) {
    int index = getCellIndex(x_index, y_index);
    this->cells[index].p_boundary = BC;
    this->cells[index].p = BC;
    this->cells[index].p_boundary_set = true;
    return 0;
}

template <class T>
int NavierStokesSolver<T>::getCellIndex(int const x_index, int const y_index) {
    if (x_index >= box_dimension_x || y_index >= box_dimension_y) {
        return -1;
    }
    return y_index * box_dimension_y + x_index;
}

template <class T>
int NavierStokesSolver<T>::getUValues(T* output) {
    for (int i = 0; i < this->box_dimension_x * this->box_dimension_y; i++) {
        output[i] = this->cells[i].u;
    }
    return 0;
}

template <class T>
int NavierStokesSolver<T>::getVValues(T* output) {
    for (int i = 0; i < this->box_dimension_x * this->box_dimension_y; i++) {
        output[i] = this->cells[i].v;
    }
    return 0;
}

template <class T>
int NavierStokesSolver<T>::getPValues(T* output) {
    for (int i = 0; i < this->box_dimension_x * this->box_dimension_y; i++) {
        output[i] = this->cells[i].p;
    }
    return 0;
}