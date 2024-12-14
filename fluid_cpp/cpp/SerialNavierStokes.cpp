//
// Created by dan on 12/10/24.
//

#include "SerialNavierStokes.h"
#include "NavierStokesCell.h"

template <class T>
SerialNavierStokes<T>::SerialNavierStokes(int box_dim_x, int box_dim_y, T domain_size_x, T domain_size_y)
    : NavierStokesSolver<T>(box_dim_x, box_dim_y, domain_size_x, domain_size_y) {

}

template <class T>
SerialNavierStokes<T>::~SerialNavierStokes() {

}

template <class T>
void SerialNavierStokes<T>::safeSolve() {
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
void SerialNavierStokes<T>::solve() {
    // loop over each time step
    for (int i = 0; i < this->num_iterations; i++) {
        this->unifiedApproximateTimeStep();
        this->unifiedComputeRightHand();

        // take a series of poisson steps to approximate the pressure in each cell
        for (int j = 0; j < this->num_poisson_iterations; j++) {
            // compute the Poisson step, enforce BCs, and enforce the pressure
            this->computePoissonStepApproximation();
            this->enforcePressureBoundaryConditions();
            this->updatePressure();
        }

        // get the pressure central difference, correct the u and v values, and enforce BCs
        this->unifiedVelocityCorrection();
        this->enforceVelocityBoundaryConditions();
    }
}

template <class T>
void SerialNavierStokes<T>::computeCentralDifference() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            // compute the central differences
            this->cells[x][y].du_dx = (this->cells[x+1][y].u - this->cells[x-1][y].u) / 2. / this->element_length_x;
            this->cells[x][y].dv_dx = (this->cells[x+1][y].v - this->cells[x-1][y].v) / 2. / this->element_length_x;
            this->cells[x][y].du_dy = (this->cells[x][y+1].u - this->cells[x][y-1].u) / 2. / this->element_length_y;
            this->cells[x][y].dv_dy = (this->cells[x][y+1].v - this->cells[x][y-1].v) / 2. / this->element_length_y;
        }
    }

}

template <class T>
void SerialNavierStokes<T>::computeLaplacian() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            // compute the laplacian
            this->cells[x][y].u_laplacian = this->cells[x-1][y].u + this->cells[x+1][y].u + this->cells[x][y+1].u + this->cells[x][y-1].u;
            this->cells[x][y].u_laplacian = (this->cells[x][y].u_laplacian - 4. * this->cells[x][y].u) / this->element_length_x / this->element_length_y;
            this->cells[x][y].v_laplacian = this->cells[x-1][y].v + this->cells[x+1][y].v + this->cells[x][y+1].v + this->cells[x][y-1].v;
            this->cells[x][y].v_laplacian = (this->cells[x][y].v_laplacian - 4. * this->cells[x][y].v) / this->element_length_x / this->element_length_y;
        }
    }
}

template <class T>
void SerialNavierStokes<T>::computeTimeDerivitive() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            // get the time derivitives
            this->cells[x][y].du_dt = this->kinematic_viscosity * this->cells[x][y].u_laplacian - this->cells[x][y].u * this->cells[x][y].du_dx - this->cells[x][y].v * this->cells[x][y].du_dy;
            this->cells[x][y].dv_dt = this->kinematic_viscosity * this->cells[x][y].v_laplacian - this->cells[x][y].u * this->cells[x][y].dv_dx - this->cells[x][y].v * this->cells[x][y].dv_dy;
        }
    }
}


template <class T>
void SerialNavierStokes<T>::takeTimeStep() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            // step forward in time
            this->cells[x][y].u_next = this->cells[x][y].u + this->time_step * this->cells[x][y].du_dt;
            this->cells[x][y].v_next = this->cells[x][y].v + this->time_step * this->cells[x][y].dv_dt;
        }
    }
}

template <class T>
void SerialNavierStokes<T>::computeNextCentralDifference() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            // compute the central differences
            this->cells[x][y].du_next_dx = (this->cells[x+1][y].u_next - this->cells[x-1][y].u_next) / 2. / this->element_length_x;
            this->cells[x][y].dv_next_dy = (this->cells[x][y+1].v_next - this->cells[x][y-1].v_next) / 2. / this->element_length_y;
        }
    }
}

template <class T>
void SerialNavierStokes<T>::computeRightHandSide() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            this->cells[x][y].right_hand_size = (this->density / this->time_step) * (this->cells[x][y].du_next_dx + this->cells[x][y].dv_next_dy);
        }
    }
}

template <class T>
void SerialNavierStokes<T>::computePoissonStepApproximation() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            // compute the Poisson step
            this->cells[x][y].p_next = this->cells[x][y].right_hand_size * this->element_length_x * this->element_length_y;
            this->cells[x][y].p_next -= this->cells[x-1][y].p + this->cells[x+1][y].p + this->cells[x][y+1].p + this->cells[x][y-1].p;
            this->cells[x][y].p_next *= -0.25;
        }
    }
}

template <class T>
void SerialNavierStokes<T>::enforcePressureBoundaryConditions() {
    // check the interior for any BC
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            if (this->cells[x][y].p_boundary_set) {
                this->cells[x][y].p_next = this->cells[x][y].p_boundary;  // enforce the BC if it has been set
            }
        }
    }

    // set the edges to be continuous with the interior
    for (int x = 0; x < this->box_dimension_x; x++) {
        if (!this->cells[x][0].p_boundary_set) {
            this->cells[x][0].p_next = this->cells[x][1].p_next;
        } else {
            this->cells[x][0].p_next = this->cells[x][0].p_boundary;
        }

        if (!this->cells[x][this->box_dimension_y - 1].p_boundary_set) {
            this->cells[x][this->box_dimension_y - 1].p_next = this->cells[x][this->box_dimension_y - 2].p_next;
        } else {
            this->cells[x][this->box_dimension_y - 1].p_next = this->cells[x][this->box_dimension_y - 1].p_boundary;
        }
    }

    for (int y = 1; y < this->box_dimension_y - 1; y++) {
        if (!this->cells[0][y].p_boundary_set) {
            this->cells[0][y].p_next = this->cells[1][y].p_next;
        } else {
            this->cells[0][y].p_next = this->cells[0][y].p_boundary;
        }
        if (!this->cells[this->box_dimension_x - 1][y].p_boundary_set) {
            this->cells[this->box_dimension_x - 1][y].p_next = this->cells[this->box_dimension_x - 2][y].p_next;
        } else {
            this->cells[this->box_dimension_x - 1][y].p_next = this->cells[this->box_dimension_x - 1][y].p_boundary;
        }
    }
}

template <class T>
void SerialNavierStokes<T>::updatePressure() {
    for (int x = 0; x < this->box_dimension_x; x++) {
        for (int y = 0; y < this->box_dimension_y; y++) {
            this->cells[x][y].p = this->cells[x][y].p_next;
        }
    }
}

template <class T>
void SerialNavierStokes<T>::computePressureCentralDifference() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            // compute the central differences
            this->cells[x][y].dp_dx = (this->cells[x+1][y].p - this->cells[x-1][y].p) / 2. / this->element_length_x;
            this->cells[x][y].dp_dy = (this->cells[x][y+1].p - this->cells[x][y-1].p) / 2. / this->element_length_y;
        }
    }
}

template <class T>
void SerialNavierStokes<T>::correctVelocityEstimates() {
    for (int x = 0; x < this->box_dimension_x; x++) {
        for (int y = 0; y < this->box_dimension_y; y++) {
            this->cells[x][y].u = this->cells[x][y].u_next - (this->time_step / this->density) * this->cells[x][y].dp_dx;
            this->cells[x][y].v = this->cells[x][y].v_next - (this->time_step / this->density) * this->cells[x][y].dp_dy;
        }
    }
}

template <class T>
void SerialNavierStokes<T>::enforceVelocityBoundaryConditions() {
    // check the interior for any BC
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            if (this->cells[x][y].u_boundary_set) {
                this->cells[x][y].u = this->cells[x][y].u_boundary;  // enforce the BC if it has been set
            }
            if (this->cells[x][y].v_boundary_set) {
                this->cells[x][y].v = this->cells[x][y].v_boundary;  // enforce the BC if it has been set
            }
        }
    }

    // set the edges to be continuous with the interior
    for (int x = 0; x < this->box_dimension_x; x++) {
        // check u bottom
        if (!this->cells[x][0].u_boundary_set) {
            this->cells[x][0].u = this->cells[x][1].u;
        } else {
            this->cells[x][0].u = this->cells[x][0].u_boundary;
        }

        // check u top
        if (!this->cells[x][this->box_dimension_y - 1].u_boundary_set) {
            this->cells[x][this->box_dimension_y - 1].u = this->cells[x][this->box_dimension_y - 2].u_next;
        } else {
            this->cells[x][this->box_dimension_y - 1].u = this->cells[x][this->box_dimension_y - 1].u_boundary;
        }

        // check v bottom
        if (!this->cells[x][0].v_boundary_set) {
            this->cells[x][0].v = this->cells[x][1].v;
        } else {
            this->cells[x][0].v = this->cells[x][0].v_boundary;
        }

        // check v top
        if (!this->cells[x][this->box_dimension_y - 1].v_boundary_set) {
            this->cells[x][this->box_dimension_y - 1].v = this->cells[x][this->box_dimension_y - 2].v_next;
        } else {
            this->cells[x][this->box_dimension_y - 1].v = this->cells[x][this->box_dimension_y - 1].v_boundary;
        }
    }

    for (int y = 1; y < this->box_dimension_y - 1; y++) {
        // check u left
        if (!this->cells[0][y].u_boundary_set) {
            this->cells[0][y].u = this->cells[1][y].u_next;
        } else {
            this->cells[0][y].u = this->cells[0][y].u_boundary;
        }

        // check u right
        if (!this->cells[this->box_dimension_x - 1][y].u_boundary_set) {
            this->cells[this->box_dimension_x - 1][y].u = this->cells[this->box_dimension_x - 2][y].u;
        } else {
            this->cells[this->box_dimension_x - 1][y].u = this->cells[this->box_dimension_x - 1][y].u_boundary;
        }

        // check v left
        if (!this->cells[0][y].v_boundary_set) {
            this->cells[0][y].v = this->cells[1][y].v_next;
        } else {
            this->cells[0][y].v = this->cells[0][y].v_boundary;
        }

        // check v right
        if (!this->cells[this->box_dimension_x - 1][y].v_boundary_set) {
            this->cells[this->box_dimension_x - 1][y].v = this->cells[this->box_dimension_x - 2][y].v;
        } else {
            this->cells[this->box_dimension_x - 1][y].v = this->cells[this->box_dimension_x - 1][y].v_boundary;
        }
    }
}

template <class T>
void SerialNavierStokes<T>::unifiedApproximateTimeStep() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            // compute the central differences
            this->cells[x][y].du_dx = (this->cells[x+1][y].u - this->cells[x-1][y].u) / 2. / this->element_length_x;
            this->cells[x][y].dv_dx = (this->cells[x+1][y].v - this->cells[x-1][y].v) / 2. / this->element_length_x;
            this->cells[x][y].du_dy = (this->cells[x][y+1].u - this->cells[x][y-1].u) / 2. / this->element_length_y;
            this->cells[x][y].dv_dy = (this->cells[x][y+1].v - this->cells[x][y-1].v) / 2. / this->element_length_y;

            // compute the laplacian
            this->cells[x][y].u_laplacian = this->cells[x-1][y].u + this->cells[x+1][y].u + this->cells[x][y+1].u + this->cells[x][y-1].u;
            this->cells[x][y].u_laplacian = (this->cells[x][y].u_laplacian - 4. * this->cells[x][y].u) / this->element_length_x / this->element_length_y;
            this->cells[x][y].v_laplacian = this->cells[x-1][y].v + this->cells[x+1][y].v + this->cells[x][y+1].v + this->cells[x][y-1].v;
            this->cells[x][y].v_laplacian = (this->cells[x][y].v_laplacian - 4. * this->cells[x][y].v) / this->element_length_x / this->element_length_y;

            // get the time derivitives
            this->cells[x][y].du_dt = this->kinematic_viscosity * this->cells[x][y].u_laplacian - this->cells[x][y].u * this->cells[x][y].du_dx - this->cells[x][y].v * this->cells[x][y].du_dy;
            this->cells[x][y].dv_dt = this->kinematic_viscosity * this->cells[x][y].v_laplacian - this->cells[x][y].u * this->cells[x][y].dv_dx - this->cells[x][y].v * this->cells[x][y].dv_dy;

            // step forward in time
            this->cells[x][y].u_next = this->cells[x][y].u + this->time_step * this->cells[x][y].du_dt;
            this->cells[x][y].v_next = this->cells[x][y].v + this->time_step * this->cells[x][y].dv_dt;
        }
    }
}

template <class T>
void SerialNavierStokes<T>::unifiedComputeRightHand() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            // compute the central differences
            this->cells[x][y].du_next_dx = (this->cells[x+1][y].u_next - this->cells[x-1][y].u_next) / 2. / this->element_length_x;
            this->cells[x][y].dv_next_dy = (this->cells[x][y+1].v_next - this->cells[x][y-1].v_next) / 2. / this->element_length_y;

            this->cells[x][y].right_hand_size = (this->density / this->time_step) * (this->cells[x][y].du_next_dx + this->cells[x][y].dv_next_dy);
        }
    }
}

template <class T>
void SerialNavierStokes<T>::unifiedVelocityCorrection() {
    for (int x = 1; x < this->box_dimension_x - 1; x++) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            // compute the central differences
            this->cells[x][y].dp_dx = (this->cells[x+1][y].p - this->cells[x-1][y].p) / 2. / this->element_length_x;
            this->cells[x][y].dp_dy = (this->cells[x][y+1].p - this->cells[x][y-1].p) / 2. / this->element_length_y;

            // compute final velocity
            this->cells[x][y].u = this->cells[x][y].u_next - (this->time_step / this->density) * this->cells[x][y].dp_dx;
            this->cells[x][y].v = this->cells[x][y].v_next - (this->time_step / this->density) * this->cells[x][y].dp_dy;
        }
    }
}
