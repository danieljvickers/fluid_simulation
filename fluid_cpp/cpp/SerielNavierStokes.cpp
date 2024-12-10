//
// Created by dan on 12/10/24.
//

#include "SerielNavierStokes.h"

template <class T>
SerielNavierStokes<T>::SerielNavierStokes(int box_dim_x, int box_dim_y, T domain_size_x, T domain_size_y)
    : NavierStokesSolver<T>(box_dim_x, box_dim_y, domain_size_x, domain_size_y) {
}

template <class T>
SerielNavierStokes<T>::~SerielNavierStokes() {

}


template <class T>
void SerielNavierStokes<T>::solve() {
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
            // compute the Poisson step, enforce BCs, and enforce the pressure
            this->computePoissonStepApproximation();
            this->enforcePressureBoundaryConditions();
            this->updatePressure();
        }

        // get the pressure central difference, correct the u and v values, and enforce BCs
        this->computePressureCentralDifference();
        this->correctVelocityEstimates();
        this->enforceVelocityBoundaryConditions();
    }
}

template <class T>
void SerielNavierStokes<T>::computeCentralDifference() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            //  get the indeices of neightboring cells
            int index = this->getCellIndex(x, y);
            int left_index = this->getCellIndex(x - 1, y);
            int right_index = this->getCellIndex(x + 1, y);
            int up_index = this->getCellIndex(x, y + 1);
            int down_index = this->getCellIndex(x, y - 1);

            // compute the central differences
            this->cells[index].du_dx = (this->cells[right_index].u - this->cells[left_index].u) / 2. / this->element_length_x;
            this->cells[index].dv_dx = (this->cells[right_index].v - this->cells[left_index].v) / 2. / this->element_length_x;
            this->cells[index].du_dy = (this->cells[up_index].u - this->cells[down_index].u) / 2. / this->element_length_y;
            this->cells[index].dv_dy = (this->cells[up_index].v - this->cells[down_index].v) / 2. / this->element_length_y;
        }
    }

}

template <class T>
void SerielNavierStokes<T>::computeLaplacian() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            //  get the indeices of neightboring cells
            int index = this->getCellIndex(x, y);
            int left_index = this->getCellIndex(x - 1, y);
            int right_index = this->getCellIndex(x + 1, y);
            int up_index = this->getCellIndex(x, y + 1);
            int down_index = this->getCellIndex(x, y - 1);

            // compute the laplacian
            this->cells[index].u_laplacian = this->cells[left_index].u + this->cells[right_index].u + this->cells[up_index].u + this->cells[down_index].u;
            this->cells[index].u_laplacian = (this->cells[index].u_laplacian - 4. * this->cells[index].u) / this->element_length_x / this->element_length_y;
            this->cells[index].v_laplacian = this->cells[left_index].v + this->cells[right_index].v + this->cells[up_index].v + this->cells[down_index].v;
            this->cells[index].v_laplacian = (this->cells[index].v_laplacian - 4. * this->cells[index].v) / this->element_length_x / this->element_length_y;
        }
    }
}

template <class T>
void SerielNavierStokes<T>::computeTimeDerivitive() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            //  get the indeix in the array of cells
            int index = this->getCellIndex(x, y);

            // get the time derivitives
            this->cells[index].du_dt = this->kinematic_viscosity * this->cells[index].u_laplacian - this->cells[index].u * this->cells[index].du_dx - this->cells[index].v * this->cells[index].du_dy;
            this->cells[index].dv_dt = this->kinematic_viscosity * this->cells[index].v_laplacian - this->cells[index].u * this->cells[index].dv_dx - this->cells[index].v * this->cells[index].dv_dy;
        }
    }
}


template <class T>
void SerielNavierStokes<T>::takeTimeStep() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            //  get the indeix in the array of cells
            int index = this->getCellIndex(x, y);

            // step forward in time
            this->cells[index].u_next = this->cells[index].u + this->time_step * this->cells[index].du_dt;
            this->cells[index].v_next = this->cells[index].v + this->time_step * this->cells[index].dv_dt;
        }
    }
}

template <class T>
void SerielNavierStokes<T>::computeNextCentralDifference() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            //  get the indeices of neightboring cells
            int index = this->getCellIndex(x, y);
            int left_index = this->getCellIndex(x - 1, y);
            int right_index = this->getCellIndex(x + 1, y);
            int up_index = this->getCellIndex(x, y + 1);
            int down_index = this->getCellIndex(x, y - 1);

            // compute the central differences
            this->cells[index].du_next_dx = (this->cells[right_index].u_next - this->cells[left_index].u_next) / 2. / this->element_length_x;
            this->cells[index].dv_next_dy = (this->cells[up_index].v_next - this->cells[down_index].v_next) / 2. / this->element_length_y;
        }
    }
}

template <class T>
void SerielNavierStokes<T>::computeRightHandSide() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            int index = this->getCellIndex(x, y);
            this->cells[index].right_hand_size = (this->density / this->time_step) * (this->cells[index].du_next_dx + this->cells[index].dv_next_dy);
        }
    }
}

template <class T>
void SerielNavierStokes<T>::computePoissonStepApproximation() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            //  get the indeices of neightboring cells
            int index = this->getCellIndex(x, y);
            int left_index = this->getCellIndex(x - 1, y);
            int right_index = this->getCellIndex(x + 1, y);
            int up_index = this->getCellIndex(x, y + 1);
            int down_index = this->getCellIndex(x, y - 1);

            // compute the Poisson step
            this->cells[index].p_next = this->cells[index].right_hand_size * this->element_length_x * this->element_length_y;
            this->cells[index].p_next -= this->cells[left_index].p + this->cells[right_index].p + this->cells[up_index].p + this->cells[down_index].p;
            this->cells[index].p_next *= -0.25;
        }
    }
}

template <class T>
void SerielNavierStokes<T>::enforcePressureBoundaryConditions() {
    // check the interior for any BC
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            int index = this->getCellIndex(x, y);
            if (this->cells[index].p_boundary_set) {
                this->cells[index].p_next = this->cells[index].p_boundary;  // enforce the BC if it has been set
            }
        }
    }

    // set the edges to be continuous with the interior
    for (int x = 0; x < this->box_dimension_x; x++) {
        if (!this->cells[this->getCellIndex(x, 0)].p_boundary_set) {
            this->cells[this->getCellIndex(x, 0)].p_next = this->cells[this->getCellIndex(x, 1)].p_next;
        } else {
            this->cells[this->getCellIndex(x, 0)].p_next = this->cells[this->getCellIndex(x, 0)].p_boundary;
        }

        if (!this->cells[this->getCellIndex(x, this->box_dimension_y - 1)].p_boundary_set) {
            this->cells[this->getCellIndex(x, this->box_dimension_y - 1)].p_next = this->cells[this->getCellIndex(x, this->box_dimension_y - 2)].p_next;
        } else {
            this->cells[this->getCellIndex(x, this->box_dimension_y - 1)].p_next = this->cells[this->getCellIndex(x, this->box_dimension_y - 1)].p_boundary;
        }
    }

    for (int y = 1; y < this->box_dimension_y - 1; y++) {
        if (!this->cells[this->getCellIndex(0, y)].p_boundary_set) {
            this->cells[this->getCellIndex(0, y)].p_next = this->cells[this->getCellIndex(1, y)].p_next;
        } else {
            this->cells[this->getCellIndex(0, y)].p_next = this->cells[this->getCellIndex(0, y)].p_boundary;
        }
        if (!this->cells[this->getCellIndex(this->box_dimension_x - 1, y)].p_boundary_set) {
            this->cells[this->getCellIndex(this->box_dimension_x - 1, y)].p_next = this->cells[this->getCellIndex(this->box_dimension_x - 2, y)].p_next;
        } else {
            this->cells[this->getCellIndex(this->box_dimension_x - 1, y)].p_next = this->cells[this->getCellIndex(this->box_dimension_x - 1, y)].p_boundary;
        }
    }
}

template <class T>
void SerielNavierStokes<T>::updatePressure() {
    for (int x = 0; x < this->box_dimension_x; x++) {
        for (int y = 0; y < this->box_dimension_y; y++) {
            int index = this->getCellIndex(x, y);
            this->cells[index].p = this->cells[index].p_next;
        }
    }
}

template <class T>
void SerielNavierStokes<T>::computePressureCentralDifference() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            //  get the indeices of neightboring cells
            int index = this->getCellIndex(x, y);
            int left_index = this->getCellIndex(x - 1, y);
            int right_index = this->getCellIndex(x + 1, y);
            int up_index = this->getCellIndex(x, y + 1);
            int down_index = this->getCellIndex(x, y - 1);

            // compute the central differences
            this->cells[index].dp_dx = (this->cells[right_index].p - this->cells[left_index].p) / 2. / this->element_length_x;
            this->cells[index].dp_dy = (this->cells[up_index].p - this->cells[down_index].p) / 2. / this->element_length_y;
        }
    }
}

template <class T>
void SerielNavierStokes<T>::correctVelocityEstimates() {
    for (int x = 0; x < this->box_dimension_x; x++) {
        for (int y = 0; y < this->box_dimension_y; y++) {
            int index = this->getCellIndex(x, y);
            this->cells[index].u = this->cells[index].u_next - (this->time_step / this->density) * this->cells[index].dp_dx;
            this->cells[index].v = this->cells[index].v_next - (this->time_step / this->density) * this->cells[index].dp_dy;
        }
    }
}

template <class T>
void SerielNavierStokes<T>::enforceVelocityBoundaryConditions() {
    // check the interior for any BC
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            int index = this->getCellIndex(x, y);
            if (this->cells[index].u_boundary_set) {
                this->cells[index].u = this->cells[index].u_boundary;  // enforce the BC if it has been set
            }
            if (this->cells[index].v_boundary_set) {
                this->cells[index].v = this->cells[index].v_boundary;  // enforce the BC if it has been set
            }
        }
    }

    // set the edges to be continuous with the interior
    for (int x = 0; x < this->box_dimension_x; x++) {
        // check u bottom
        if (!this->cells[this->getCellIndex(x, 0)].u_boundary_set) {
            this->cells[this->getCellIndex(x, 0)].u = this->cells[this->getCellIndex(x, 1)].u;
        } else {
            this->cells[this->getCellIndex(x, 0)].u = this->cells[this->getCellIndex(x, 0)].u_boundary;
        }

        // check u top
        if (!this->cells[this->getCellIndex(x, this->box_dimension_y - 1)].u_boundary_set) {
            this->cells[this->getCellIndex(x, this->box_dimension_y - 1)].u = this->cells[this->getCellIndex(x, this->box_dimension_y - 2)].u_next;
        } else {
            this->cells[this->getCellIndex(x, this->box_dimension_y - 1)].u = this->cells[this->getCellIndex(x, this->box_dimension_y - 1)].u_boundary;
        }

        // check v bottom
        if (!this->cells[this->getCellIndex(x, 0)].v_boundary_set) {
            this->cells[this->getCellIndex(x, 0)].v = this->cells[this->getCellIndex(x, 1)].v;
        } else {
            this->cells[this->getCellIndex(x, 0)].v = this->cells[this->getCellIndex(x, 0)].v_boundary;
        }

        // check v top
        if (!this->cells[this->getCellIndex(x, this->box_dimension_y - 1)].v_boundary_set) {
            this->cells[this->getCellIndex(x, this->box_dimension_y - 1)].v = this->cells[this->getCellIndex(x, this->box_dimension_y - 2)].v_next;
        } else {
            this->cells[this->getCellIndex(x, this->box_dimension_y - 1)].v = this->cells[this->getCellIndex(x, this->box_dimension_y - 1)].v_boundary;
        }
    }

    for (int y = 1; y < this->box_dimension_y - 1; y++) {
        // check u left
        if (!this->cells[this->getCellIndex(0, y)].u_boundary_set) {
            this->cells[this->getCellIndex(0, y)].u = this->cells[this->getCellIndex(1, y)].u_next;
        } else {
            this->cells[this->getCellIndex(0, y)].u = this->cells[this->getCellIndex(0, y)].u_boundary;
        }

        // check u right
        if (!this->cells[this->getCellIndex(this->box_dimension_x - 1, y)].u_boundary_set) {
            this->cells[this->getCellIndex(this->box_dimension_x - 1, y)].u = this->cells[this->getCellIndex(this->box_dimension_x - 2, y)].u;
        } else {
            this->cells[this->getCellIndex(this->box_dimension_x - 1, y)].u = this->cells[this->getCellIndex(this->box_dimension_x - 1, y)].u_boundary;
        }

        // check v left
        if (!this->cells[this->getCellIndex(0, y)].v_boundary_set) {
            this->cells[this->getCellIndex(0, y)].v = this->cells[this->getCellIndex(1, y)].v_next;
        } else {
            this->cells[this->getCellIndex(0, y)].v = this->cells[this->getCellIndex(0, y)].v_boundary;
        }

        // check v right
        if (!this->cells[this->getCellIndex(this->box_dimension_x - 1, y)].v_boundary_set) {
            this->cells[this->getCellIndex(this->box_dimension_x - 1, y)].v = this->cells[this->getCellIndex(this->box_dimension_x - 2, y)].v;
        } else {
            this->cells[this->getCellIndex(this->box_dimension_x - 1, y)].v = this->cells[this->getCellIndex(this->box_dimension_x - 1, y)].v_boundary;
        }
    }
}
