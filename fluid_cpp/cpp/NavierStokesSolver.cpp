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
void NavierStokesSolver<T>::solve() {
    // loop over each time step
    for (int i = 0; i < this->num_iterations; i++) {
        // compute useful derivatives
        this->computeCentralDifference();
        this->computeLaplacian();
        this->computeTimeDerivitive();

        // take a tenative step forward in time
        this->takeTimeStep();
        this->computeNextCentralDifference(); // recompute the central difference
        this->computeRightHandSide();

        // take a series of poisson steps to approximate the pressure in each cell
        for (int j = 0; j < this->num_poisson_iterations; j++) {
            this->computePoissonStepApproximation();

            this->enforcePressureBoundaryConditions();
            this->updatePressure();
        }

        // get the pressure central difference, and set the u and v values
        this->computePressureCentralDifference();

        // get the pressure central difference, and set the u and v values
        this->correctVelocityEstimates();

        this->enforceVelocityBoundaryConditions();
    }
}

template <class T>
int NavierStokesSolver<T>::getCellIndex(int const x_index, int const y_index) {
    if (x_index >= box_dimension_x || y_index >= box_dimension_y) {
        return -1;
    }
    return y_index * box_dimension_y + x_index;
}

template <class T>
void NavierStokesSolver<T>::computeCentralDifference() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            //  get the indeices of neightboring cells
            int index = getCellIndex(x, y);
            int left_index = getCellIndex(x - 1, y);
            int right_index = getCellIndex(x + 1, y);
            int up_index = getCellIndex(x, y + 1);
            int down_index = getCellIndex(x, y - 1);

            // compute the central differences
            cells[index].du_dx = (cells[right_index].u - cells[left_index].u) / 2. / element_length_x;
            cells[index].dv_dx = (cells[right_index].v - cells[left_index].v) / 2. / element_length_x;
            cells[index].du_dy = (cells[up_index].u - cells[down_index].u) / 2. / element_length_y;
            cells[index].dv_dy = (cells[up_index].v - cells[down_index].v) / 2. / element_length_y;
        }
    }

}

template <class T>
void NavierStokesSolver<T>::computeLaplacian() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            //  get the indeices of neightboring cells
            int index = getCellIndex(x, y);
            int left_index = getCellIndex(x - 1, y);
            int right_index = getCellIndex(x + 1, y);
            int up_index = getCellIndex(x, y + 1);
            int down_index = getCellIndex(x, y - 1);

            // compute the laplacian
            cells[index].u_laplacian = cells[left_index].u + cells[right_index].u + cells[up_index].u + cells[down_index].u;
            cells[index].u_laplacian = (cells[index].u_laplacian - 4. * cells[index].u) / element_length_x / element_length_y;
            cells[index].v_laplacian = cells[left_index].v + cells[right_index].v + cells[up_index].v + cells[down_index].v;
            cells[index].v_laplacian = (cells[index].v_laplacian - 4. * cells[index].v) / element_length_x / element_length_y;
        }
    }
}

template <class T>
void NavierStokesSolver<T>::computeTimeDerivitive() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            //  get the indeix in the array of cells
            int index = getCellIndex(x, y);

            // get the time derivitives
            cells[index].du_dt = kinematic_viscosity * cells[index].u_laplacian - cells[index].u * cells[index].du_dx - cells[index].v * cells[index].du_dy;
            cells[index].dv_dt = kinematic_viscosity * cells[index].v_laplacian - cells[index].u * cells[index].dv_dx - cells[index].v * cells[index].dv_dy;
        }
    }
}


template <class T>
void NavierStokesSolver<T>::takeTimeStep() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            //  get the indeix in the array of cells
            int index = getCellIndex(x, y);

            // step forward in time
            cells[index].u_next = cells[index].u + time_step * cells[index].du_dt;
            cells[index].v_next = cells[index].v + time_step * cells[index].dv_dt;
        }
    }
}

template <class T>
void NavierStokesSolver<T>::computeNextCentralDifference() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            //  get the indeices of neightboring cells
            int index = getCellIndex(x, y);
            int left_index = getCellIndex(x - 1, y);
            int right_index = getCellIndex(x + 1, y);
            int up_index = getCellIndex(x, y + 1);
            int down_index = getCellIndex(x, y - 1);

            // compute the central differences
            cells[index].du_next_dx = (cells[right_index].u_next - cells[left_index].u_next) / 2. / element_length_x;
            cells[index].dv_next_dy = (cells[up_index].v_next - cells[down_index].v_next) / 2. / element_length_y;
        }
    }
}

template <class T>
void NavierStokesSolver<T>::computeRightHandSide() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            int index = getCellIndex(x, y);
            cells[index].right_hand_size = (density / time_step) * (cells[index].du_next_dx + cells[index].dv_next_dy);
        }
    }
}

template <class T>
void NavierStokesSolver<T>::computePoissonStepApproximation() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            //  get the indeices of neightboring cells
            int index = getCellIndex(x, y);
            int left_index = getCellIndex(x - 1, y);
            int right_index = getCellIndex(x + 1, y);
            int up_index = getCellIndex(x, y + 1);
            int down_index = getCellIndex(x, y - 1);

            // compute the Poisson step
            cells[index].p_next = cells[index].right_hand_size * element_length_x * element_length_y;
            cells[index].p_next -= cells[left_index].p + cells[right_index].p + cells[up_index].p + cells[down_index].p;
            cells[index].p_next *= -0.25;
        }
    }
}

template <class T>
void NavierStokesSolver<T>::enforcePressureBoundaryConditions() {
    // check the interior for any BC
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            int index = this->getCellIndex(x, y);
            if (this->cells[index].p_boundary_set) {
                cells[index].p_next = this->cells[index].p_boundary;  // enforce the BC if it has been set
            }
        }
    }

    // set the edges to be continuous with the interior
    for (int x = 0; x < box_dimension_x; x++) {
        if (!cells[getCellIndex(x, 0)].p_boundary_set) {
            cells[getCellIndex(x, 0)].p_next = cells[getCellIndex(x, 1)].p_next;
        } else {
            cells[getCellIndex(x, 0)].p_next = cells[getCellIndex(x, 0)].p_boundary;
        }

        if (!cells[getCellIndex(x, box_dimension_y - 1)].p_boundary_set) {
            cells[getCellIndex(x, box_dimension_y - 1)].p_next = cells[getCellIndex(x, box_dimension_y - 2)].p_next;
        } else {
            cells[getCellIndex(x, box_dimension_y - 1)].p_next = cells[getCellIndex(x, box_dimension_y - 1)].p_boundary;
        }
    }

    for (int y = 1; y < box_dimension_y - 1; y++) {
        if (!cells[getCellIndex(0, y)].p_boundary_set) {
            cells[getCellIndex(0, y)].p_next = cells[getCellIndex(1, y)].p_next;
        } else {
            cells[getCellIndex(0, y)].p_next = cells[getCellIndex(0, y)].p_boundary;
        }
        if (!cells[getCellIndex(box_dimension_x - 1, y)].p_boundary_set) {
            cells[getCellIndex(box_dimension_x - 1, y)].p_next = cells[getCellIndex(box_dimension_x - 2, y)].p_next;
        } else {
            cells[getCellIndex(box_dimension_x - 1, y)].p_next = cells[getCellIndex(box_dimension_x - 1, y)].p_boundary;
        }
    }
}

template <class T>
void NavierStokesSolver<T>::updatePressure() {
    for (int x = 0; x < box_dimension_x; x++) {
        for (int y = 0; y < box_dimension_y; y++) {
            int index = getCellIndex(x, y);
            cells[index].p = cells[index].p_next;
        }
    }
}

template <class T>
void NavierStokesSolver<T>::computePressureCentralDifference() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            //  get the indeices of neightboring cells
            int index = getCellIndex(x, y);
            int left_index = getCellIndex(x - 1, y);
            int right_index = getCellIndex(x + 1, y);
            int up_index = getCellIndex(x, y + 1);
            int down_index = getCellIndex(x, y - 1);

            // compute the central differences
            cells[index].dp_dx = (cells[right_index].p - cells[left_index].p) / 2. / element_length_x;
            cells[index].dp_dy = (cells[up_index].p - cells[down_index].p) / 2. / element_length_y;
        }
    }
}

template <class T>
void NavierStokesSolver<T>::correctVelocityEstimates() {
    for (int x = 0; x < box_dimension_x; x++) {
        for (int y = 0; y < box_dimension_y; y++) {
            int index = getCellIndex(x, y);
            cells[index].u = cells[index].u_next - (time_step / density) * cells[index].dp_dx;
            cells[index].v = cells[index].v_next - (time_step / density) * cells[index].dp_dy;
        }
    }
}

template <class T>
void NavierStokesSolver<T>::enforceVelocityBoundaryConditions() {
    // check the interior for any BC
    for (int x = 1; x < box_dimension_x - 1; x++) {
        for (int y = 1; y < box_dimension_y - 1; y++) {
            int index = getCellIndex(x, y);
            if (cells[index].u_boundary_set) {
                cells[index].u = cells[index].u_boundary;  // enforce the BC if it has been set
            }
            if (cells[index].v_boundary_set) {
                cells[index].v = cells[index].v_boundary;  // enforce the BC if it has been set
            }
        }
    }

    // set the edges to be continuous with the interior
    for (int x = 0; x < box_dimension_x; x++) {
        // check u bottom
        if (!cells[getCellIndex(x, 0)].u_boundary_set) {
            cells[getCellIndex(x, 0)].u = cells[getCellIndex(x, 1)].u;
        } else {
            cells[getCellIndex(x, 0)].u = cells[getCellIndex(x, 0)].u_boundary;
        }

        // check u top
        if (!cells[getCellIndex(x, box_dimension_y - 1)].u_boundary_set) {
            cells[getCellIndex(x, box_dimension_y - 1)].u = cells[getCellIndex(x, box_dimension_y - 2)].u_next;
        } else {
            cells[getCellIndex(x, box_dimension_y - 1)].u = cells[getCellIndex(x, box_dimension_y - 1)].u_boundary;
        }

        // check v bottom
        if (!cells[getCellIndex(x, 0)].v_boundary_set) {
            cells[getCellIndex(x, 0)].v = cells[getCellIndex(x, 1)].v;
        } else {
            cells[getCellIndex(x, 0)].v = cells[getCellIndex(x, 0)].v_boundary;
        }

        // check v top
        if (!cells[getCellIndex(x, box_dimension_y - 1)].v_boundary_set) {
            cells[getCellIndex(x, box_dimension_y - 1)].v = cells[getCellIndex(x, box_dimension_y - 2)].v_next;
        } else {
            cells[getCellIndex(x, box_dimension_y - 1)].v = cells[getCellIndex(x, box_dimension_y - 1)].v_boundary;
        }
    }

    for (int y = 1; y < box_dimension_y - 1; y++) {
        // check u left
        if (!cells[getCellIndex(0, y)].u_boundary_set) {
            cells[getCellIndex(0, y)].u = cells[getCellIndex(1, y)].u_next;
        } else {
            cells[getCellIndex(0, y)].u = cells[getCellIndex(0, y)].u_boundary;
        }

        // check u right
        if (!cells[getCellIndex(box_dimension_x - 1, y)].u_boundary_set) {
            cells[getCellIndex(box_dimension_x - 1, y)].u = cells[getCellIndex(box_dimension_x - 2, y)].u;
        } else {
            cells[getCellIndex(box_dimension_x - 1, y)].u = cells[getCellIndex(box_dimension_x - 1, y)].u_boundary;
        }

        // check v left
        if (!cells[getCellIndex(0, y)].v_boundary_set) {
            cells[getCellIndex(0, y)].v = cells[getCellIndex(1, y)].v_next;
        } else {
            cells[getCellIndex(0, y)].v = cells[getCellIndex(0, y)].v_boundary;
        }

        // check v right
        if (!cells[getCellIndex(box_dimension_x - 1, y)].v_boundary_set) {
            cells[getCellIndex(box_dimension_x - 1, y)].v = cells[getCellIndex(box_dimension_x - 2, y)].v;
        } else {
            cells[getCellIndex(box_dimension_x - 1, y)].v = cells[getCellIndex(box_dimension_x - 1, y)].v_boundary;
        }
    }
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