//
// Created by Daniel J. Vickers on 11/12/24.
//

#include "NavierStokesSolver.h"
#include "NavierStokesCell.h"

#include <ios>
#include <iostream>

template <class T>
NavierStokesSolver<T>::NavierStokesSolver(int box_dim_x, int box_dim_y, T domain_size_x, T domain_size_y) {
    box_dimension_x = box_dim_x;
    box_dimension_y = box_dim_y;
    this->cells = new NavierStokesCell<T>*[box_dim_x];
    for (int x = 0; x < box_dim_x; x++) {
        this->cells[x] = new NavierStokesCell<T>[box_dim_y];
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

    for (int x = 0; x < box_dimension_x; x++) {
        free(this->cells[x]);
    }
    free(this->cells);
    this->box_dimension_x = box_dimension_x;
    this->box_dimension_y = box_dimension_y;
    this->cells = new NavierStokesCell<T>*[box_dimension_x];
    for (int x = 0; x < box_dimension_x; x++) {
        this->cells[x] = new NavierStokesCell<T>[box_dimension_y];
    }
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
int NavierStokesSolver<T>::setUBoundaryCondition(int const x, int const y, T const BC) {
    this->cells[x][y].u_boundary = BC;
    this->cells[x][y].u = BC;
    this->cells[x][y].u_boundary_set = true;
    return 0;
}

template <class T>
int NavierStokesSolver<T>::setVBoundaryCondition(int const x, int const y, T const BC) {
    this->cells[x][y].v_boundary = BC;
    this->cells[x][y].v = BC;
    this->cells[x][y].v_boundary_set = true;
    return 0;
}

template <class T>
int NavierStokesSolver<T>::setPBoundaryCondition(int const x, int const y, T const BC) {
    this->cells[x][y].p_boundary = BC;
    this->cells[x][y].p = BC;
    this->cells[x][y].p_boundary_set = true;
    return 0;
}

template <class T>
int NavierStokesSolver<T>::getUValues(T* output) {
    for (int x = 0; x < this->box_dimension_x; x++) {
        for (int y = 0; y <  this->box_dimension_y; y++) {
            output[x * this->box_dimension_x + y] = this->cells[x][y].u;
        }
    }
    return 0;
}

template <class T>
int NavierStokesSolver<T>::getVValues(T* output) {
    for (int x = 0; x < this->box_dimension_x; x++) {
        for (int y = 0; y <  this->box_dimension_y; y++) {
            output[x * this->box_dimension_x + y] = this->cells[x][y].v;
        }
    }
    return 0;
}

template <class T>
int NavierStokesSolver<T>::getPValues(T* output) {
    for (int x = 0; x < this->box_dimension_x; x++) {
        for (int y = 0; y <  this->box_dimension_y; y++) {
            output[x * this->box_dimension_x + y] = this->cells[x][y].p;
        }
    }
    return 0;
}