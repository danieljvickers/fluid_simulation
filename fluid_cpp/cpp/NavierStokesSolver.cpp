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
    this->cells = static_cast<NavierStokesCell<T>*>(malloc(sizeof(NavierStokesCell<T>) * box_dimension_x * box_dimension_y));

    this->setDomainSize(domain_size_x, domain_size_x);
}

template <class T>
NavierStokesSolver<T>::~NavierStokesSolver() {
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
    return 0;
}

template <class T>
int NavierStokesSolver<T>::setVBoundaryCondition(int const x_index, int const y_index, T const BC) {
    int index = getCellIndex(x_index, y_index);
    this->cells[index].v_boundary = BC;
    this->cells[index].v = BC;
    return 0;
}

template <class T>
int NavierStokesSolver<T>::setPBoundaryCondition(int const x_index, int const y_index, T const BC) {
    int index = getCellIndex(x_index, y_index);
    this->cells[index].p_boundary = BC;
    this->cells[index].p = BC;
    return 0;
}

template <class T>
void NavierStokesSolver<T>::solve() {
    // loop over each time step
    for (int i = 0; i < num_iterations; i++) {
        // loop over every cell in the space
        for (int x = 1; x < box_dimension_x - 1; x++) {
            for (int y = 1; y < box_dimension_y - 1; y++) {
                // compute the initial derivates for u and v
                computeCentralDifference(x, y);
                computeLaplacian(x, y);
                computeTimeDerivitive(x, y);

                // take a tenative step forward in time
                takeTimeStep(x, y);
                computeNextCentralDifference(x, y); // recompute the central difference
                computeRightHandSide(x, y);
            }
        }
    }

    // take a series of poisson steps to approximate the pressure in each cell
    for (int j = 0; j < num_poisson_iterations; j++) {
        for (int x = 1; x < box_dimension_x - 1; x++) {
            for (int y = 1; y < box_dimension_y - 1; y++) {
                computePoissonStepApproximation(x, y);
            }
        }
        enforcePressureBoundaryConditions();
        updatePressure();
    }

    // get the pressure central difference, and set the u and v values
    for (int x = 1; x < box_dimension_x - 1; x++) {
        for (int y = 1; y < box_dimension_y - 1; y++) {
            computePressureCentralDifference(x, y);
            correctVelocityEstimates(x, y);
        }
    }

    enforceVelocityBoundaryConditions();
}

template <class T>
int NavierStokesSolver<T>::getCellIndex(int const x_index, int const y_index) {
    if (x_index >= box_dimension_x || y_index >= box_dimension_y) {
        return -1;
    }
    return y_index * box_dimension_y + x_index;
}

template <class T>
void NavierStokesSolver<T>::computeCentralDifference(int const index_x, int const index_y) {
    //  get the indeices of neightboring cells
    int cell_index = getCellIndex(index_x, index_y);
    int left_index = getCellIndex(index_x - 1, index_y);
    int right_index = getCellIndex(index_x + 1, index_y);
    int up_index = getCellIndex(index_x, index_y + 1);
    int down_index = getCellIndex(index_x, index_y - 1);

    // compute the central differences
    cells[cell_index].du_dx = (cells[right_index].u - cells[left_index].u) / 2. / element_length_x;
    cells[cell_index].dv_dx = (cells[right_index].v - cells[left_index].v) / 2. / element_length_x;
    cells[cell_index].du_dy = (cells[up_index].u - cells[down_index].u) / 2. / element_length_y;
    cells[cell_index].dv_dy = (cells[up_index].v - cells[down_index].v) / 2. / element_length_y;
}

template <class T>
void NavierStokesSolver<T>::computeLaplacian(int const index_x, int const index_y) {
    //  get the indeices of neightboring cells
    int index = getCellIndex(index_x, index_y);
    int left_index = getCellIndex(index_x - 1, index_y);
    int right_index = getCellIndex(index_x + 1, index_y);
    int up_index = getCellIndex(index_x, index_y + 1);
    int down_index = getCellIndex(index_x, index_y - 1);

    // compute the laplacian
    cells[index].u_laplacian = cells[left_index].u + cells[right_index].u + cells[up_index].u + cells[down_index].u;
    cells[index].u_laplacian = (cells[index].u_laplacian - 4. * cells[index].u) / element_length_x / element_length_y;
    cells[index].v_laplacian = cells[left_index].v + cells[right_index].v + cells[up_index].v + cells[down_index].v;
    cells[index].v_laplacian = (cells[index].v_laplacian - 4. * cells[index].v) / element_length_x / element_length_y;
}

template <class T>
void NavierStokesSolver<T>::computeTimeDerivitive(int const index_x, int const index_y) {
    //  get the indeices of neightboring cells
    int index = getCellIndex(index_x, index_y);

    // get the time derivitives
    cells[index].du_dt = kinematic_viscosity * cells[index].u_laplacian - cells[index].u * cells[index].du_dx - cells[index].v * cells[index].du_dy;
    cells[index].du_dt = kinematic_viscosity * cells[index].u_laplacian - cells[index].u * cells[index].du_dx - cells[index].v * cells[index].du_dy;
}

template <class T>
void NavierStokesSolver<T>::takeTimeStep(int const index_x, int const index_y) {
    //  get the indeices of neightboring cells
    int index = getCellIndex(index_x, index_y);

    // get the time derivitives
    cells[index].u_next = cells[index].u + time_step * cells[index].du_dt;
    cells[index].v_next = cells[index].v + time_step * cells[index].dv_dt;
}

template <class T>
void NavierStokesSolver<T>::computeNextCentralDifference(int const index_x, int const index_y) {
    //  get the indeices of neightboring cells
    int index = getCellIndex(index_x, index_y);
    int left_index = getCellIndex(index_x - 1, index_y);
    int right_index = getCellIndex(index_x + 1, index_y);
    int up_index = getCellIndex(index_x, index_y + 1);
    int down_index = getCellIndex(index_x, index_y - 1);

    // compute the central differences
    cells[index].du_next_dx = (cells[right_index].u_next - cells[left_index].u_next) / 2. / element_length_x;
    cells[index].dv_next_dy = (cells[up_index].v_next - cells[down_index].v_next) / 2. / element_length_y;

}

template <class T>
void NavierStokesSolver<T>::computeRightHandSide(int const index_x, int const index_y) {
    int index = getCellIndex(index_x, index_y);
    cells[index].right_hand_size = (density / time_step) * (cells[index].du_next_dx + cells[index].dv_next_dy);
}

template <class T>
void NavierStokesSolver<T>::computePoissonStepApproximation(int const index_x, int const index_y) {
    int index = getCellIndex(index_x, index_y);
    int left_index = getCellIndex(index_x - 1, index_y);
    int right_index = getCellIndex(index_x + 1, index_y);
    int up_index = getCellIndex(index_x, index_y + 1);
    int down_index = getCellIndex(index_x, index_y - 1);

    cells[index].p_next = cells[index].right_hand_size * element_length_x * element_length_y;
    cells[index].p_next -= cells[left_index].p + cells[right_index].p + cells[up_index].p + cells[down_index].p;
    cells[index].p_next *= -0.25;
}

template <class T>
void NavierStokesSolver<T>::enforcePressureBoundaryConditions() {
    // check the interior for any BC
    for (int x = 1; x < domain_size_x - 1; x++) {
        for (int y = 1; y < domain_size_y - 1; y++) {
            int index = getCellIndex(x, y);
            if (!std::isnan(cells[index].p_boundary)) {
                cells[index].p_next = cells[index].p_boundary;  // enforce the BC if it has been set
            }
        }
    }

    // set the edges to be continuous with the interior
    for (int x = 0; x < domain_size_x; x++) {
        if (std::isnan(cells[getCellIndex(x, 0)].p_boundary)) {
            cells[getCellIndex(x, 0)].p_next = cells[getCellIndex(x, 1)].p_next;
        } else {
            cells[getCellIndex(x, 0)].p_next = cells[getCellIndex(x, 0)].p_boundary;
        }

        if (std::isnan(cells[getCellIndex(x, domain_size_y - 1)].p_boundary)) {
            cells[getCellIndex(x, domain_size_y - 1)].p_next = cells[getCellIndex(x, domain_size_y - 2)].p_next;
        } else {
            cells[getCellIndex(x, domain_size_y - 1)].p_next = cells[getCellIndex(x, domain_size_y - 1)].p_boundary;
        }
    }

    for (int y = 1; y < domain_size_y - 1; y++) {
        if (std::isnan(cells[getCellIndex(0, y)].p_boundary)) {
            cells[getCellIndex(0, y)].p_next = cells[getCellIndex(1, y)].p_next;
        } else {
            cells[getCellIndex(0, y)].p_next = cells[getCellIndex(0, y)].p_boundary;
        }
        if (std::isnan(cells[getCellIndex(domain_size_x - 1, y)].p_boundary)) {
            cells[getCellIndex(domain_size_x - 1, y)].p_next = cells[getCellIndex(domain_size_x - 2, y)].p_next;
        } else {
            cells[getCellIndex(domain_size_x - 1, y)].p_next = cells[getCellIndex(domain_size_x - 1, y)].p_boundary;
        }
    }
}

template <class T>
void NavierStokesSolver<T>::updatePressure() {
    // check the interior for any BC
    for (int x = 0; x < domain_size_x - 1; x++) {
        for (int y = 1; y < domain_size_y - 1; y++) {
            int index = getCellIndex(x, y);
            cells[index].p = cells[index].p_next;
        }
    }
}

template <class T>
void NavierStokesSolver<T>::computePressureCentralDifference(int const index_x, int const index_y) {
    //  get the indeices of neightboring cells
    int cell_index = getCellIndex(index_x, index_y);
    int left_index = getCellIndex(index_x - 1, index_y);
    int right_index = getCellIndex(index_x + 1, index_y);
    int up_index = getCellIndex(index_x, index_y + 1);
    int down_index = getCellIndex(index_x, index_y - 1);

    // compute the central differences
    cells[cell_index].dp_dx = (cells[right_index].p - cells[left_index].p) / 2. / element_length_x;
    cells[cell_index].dp_dy = (cells[up_index].p - cells[down_index].p) / 2. / element_length_y;
}

template <class T>
void NavierStokesSolver<T>::correctVelocityEstimates(int const index_x, int const index_y) {
    //  get the indeices of neightboring cells
    int index = getCellIndex(index_x, index_y);

    // get the time derivitives
    cells[index].u = cells[index].u_next - (time_step / density) * cells[index].dp_dx;
    cells[index].v = cells[index].v_next - (time_step / density) * cells[index].dp_dy;
}

template <class T>
void NavierStokesSolver<T>::enforceVelocityBoundaryConditions() {
    // check the interior for any BC
    for (int x = 1; x < domain_size_x - 1; x++) {
        for (int y = 1; y < domain_size_y - 1; y++) {
            int index = getCellIndex(x, y);
            if (!std::isnan(cells[index].u_boundary)) {
                cells[index].u = cells[index].u_boundary;  // enforce the BC if it has been set
            }
            if (!std::isnan(cells[index].v_boundary)) {
                cells[index].v = cells[index].v_boundary;  // enforce the BC if it has been set
            }
        }
    }

    // set the edges to be continuous with the interior
    for (int x = 0; x < domain_size_x; x++) {
        // check u bottom
        if (std::isnan(cells[getCellIndex(x, 0)].u_boundary)) {
            cells[getCellIndex(x, 0)].u = cells[getCellIndex(x, 1)].u;
        } else {
            cells[getCellIndex(x, 0)].u = cells[getCellIndex(x, 0)].u_boundary;
        }

        // check u top
        if (std::isnan(cells[getCellIndex(x, domain_size_y - 1)].u_boundary)) {
            cells[getCellIndex(x, domain_size_y - 1)].u = cells[getCellIndex(x, domain_size_y - 2)].u_next;
        } else {
            cells[getCellIndex(x, domain_size_y - 1)].u = cells[getCellIndex(x, domain_size_y - 1)].u_boundary;
        }

        // check v bottom
        if (std::isnan(cells[getCellIndex(x, 0)].v_boundary)) {
            cells[getCellIndex(x, 0)].v = cells[getCellIndex(x, 1)].v;
        } else {
            cells[getCellIndex(x, 0)].v = cells[getCellIndex(x, 0)].v_boundary;
        }

        // check v top
        if (std::isnan(cells[getCellIndex(x, domain_size_y - 1)].v_boundary)) {
            cells[getCellIndex(x, domain_size_y - 1)].v = cells[getCellIndex(x, domain_size_y - 2)].v_next;
        } else {
            cells[getCellIndex(x, domain_size_y - 1)].v = cells[getCellIndex(x, domain_size_y - 1)].v_boundary;
        }
    }

    for (int y = 1; y < domain_size_y - 1; y++) {
        // check u left
        if (std::isnan(cells[getCellIndex(0, y)].u_boundary)) {
            cells[getCellIndex(0, y)].u = cells[getCellIndex(1, y)].u_next;
        } else {
            cells[getCellIndex(0, y)].u = cells[getCellIndex(0, y)].u_boundary;
        }

        // check u right
        if (std::isnan(cells[getCellIndex(domain_size_x - 1, y)].u_boundary)) {
            cells[getCellIndex(domain_size_x - 1, y)].u = cells[getCellIndex(domain_size_x - 2, y)].u;
        } else {
            cells[getCellIndex(domain_size_x - 1, y)].u = cells[getCellIndex(domain_size_x - 1, y)].u_boundary;
        }

        // check v left
        if (std::isnan(cells[getCellIndex(0, y)].v_boundary)) {
            cells[getCellIndex(0, y)].v = cells[getCellIndex(1, y)].v_next;
        } else {
            cells[getCellIndex(0, y)].v = cells[getCellIndex(0, y)].v_boundary;
        }

        // check v right
        if (std::isnan(cells[getCellIndex(domain_size_x - 1, y)].v_boundary)) {
            cells[getCellIndex(domain_size_x - 1, y)].v = cells[getCellIndex(domain_size_x - 2, y)].v;
        } else {
            cells[getCellIndex(domain_size_x - 1, y)].v = cells[getCellIndex(domain_size_x - 1, y)].v_boundary;
        }
    }
}

template <class T>
int NavierStokesSolver<T>::getUValues(T* output) {
    for (int i = 0; i < this->box_dimension_x * this->box_dimension_y; i++) {
        output[i] = this->cells[i].u;
        std::cout << "U Values: " << output[i] << " :: " << this->cells[i].u << std::endl;
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