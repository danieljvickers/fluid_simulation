//
// Created by dan on 12/18/24.
//

#include "ThreadedNavierStokes.h"
#include <thread>
#include <shared_mutex>


template <class T>
ThreadedNavierStokes<T>::ThreadedNavierStokes(int box_dim_x, int box_dim_y, T domain_size_x, T domain_size_y)
    : NavierStokesSolver<T>(box_dim_x, box_dim_y, domain_size_x, domain_size_y) {
    this->worker_threads = new std::thread[THREADED_GRID_SIZE];
}

template <class T>
ThreadedNavierStokes<T>::~ThreadedNavierStokes() {

}

template <class T>
void ThreadedNavierStokes<T>::syncThreads() {
    this->sync_mutex.lock(); // aquire the lock
    if (this->lock_count >= THREADED_GRID_SIZE) this->lock_count = 0;  // reset the counter if this thread is the first to arrive
    this->lock_count += 1; // incriment the counter
    this->sync_mutex.unlock();  // release the mutex to other threads

    while(this->lock_count < THREADED_GRID_SIZE) {}  // block the thread until all threads have checked in
}

template <class T>
void ThreadedNavierStokes<T>::solveThread(int index) {
    // loop over each time step
    for (int i = 0; i < this->num_iterations; i++) {
        this->unifiedApproximateTimeStep(index);
        this->syncThreads();

        this->unifiedComputeRightHand(index);
        this->syncThreads();

        // take a series of poisson steps to approximate the pressure in each cell
        for (int j = 0; j < this->num_poisson_iterations; j++) {
            this->computePoissonStepApproximation(index);
            this->syncThreads();

            this->enforcePressureBoundaryConditions(index);
            // this->syncThreads();

            this->updatePressure(index);
            this->syncThreads();
        }

        this->unifiedVelocityCorrection(index);
        this->syncThreads();

        this->enforceVelocityBoundaryConditions(index);
        this->syncThreads();
    }
}

template <class T>
void ThreadedNavierStokes<T>::solve() {
    for (int t = 0; t < THREADED_GRID_SIZE; t++) {
        // begin a thread to work on the unified time step approximation functions
        this->worker_threads[t] = std::thread(&ThreadedNavierStokes::solveThread, this, t);
    }
    for (int t = 0; t < THREADED_GRID_SIZE; t++) {
        // join all of the worker threads when they are complete
        this->worker_threads[t].join();
    }
}

template <class T>
void ThreadedNavierStokes<T>::computePoissonStepApproximation(int thread_index) {
    int start_x = thread_index + 1;

    for (int x = start_x; x < this->box_dimension_x - 1; x += THREADED_GRID_SIZE) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            // compute the Poisson step
            this->cells[x][y].p_next = this->cells[x][y].right_hand_size * this->element_length_x * this->element_length_y;
            this->cells[x][y].p_next -= this->cells[x-1][y].p + this->cells[x+1][y].p + this->cells[x][y+1].p + this->cells[x][y-1].p;
            this->cells[x][y].p_next *= -0.25;
        }
    }
}

template <class T>
void ThreadedNavierStokes<T>::enforcePressureBoundaryConditions(int thread_index) {
    int start_x = thread_index;

    for (int x = start_x; x < this->box_dimension_x; x += THREADED_GRID_SIZE) {
        for (int y = 0; y < this->box_dimension_y ; y++) {
            if (this->cells[x][y].p_boundary_set) {
                this->cells[x][y].p_next = this->cells[x][y].p_boundary;  // enforce the BC if it has been set
                continue;
            }

            // checks if you are on the edge, else do nothing
            if (y == 0) {
                this->cells[x][y].p_next = this->cells[x][y+1].p_next; // equal to cell above
            } else if (y == this->box_dimension_y - 1) {
                this->cells[x][y].p_next = this->cells[x][y-1].p_next; // equal to cell below
            } else if (x == 0) {
                this->cells[x][y].p_next = this->cells[x+1][y].p_next; // equal to cell to right
            } else if (x == this->box_dimension_x - 1) {
                this->cells[x][y].p_next = this->cells[x-1][y].p_next;  // equal to cell to left
            }
        }
    }
}

template <class T>
void ThreadedNavierStokes<T>::updatePressure(int thread_index) {
    int start_x = thread_index;

    for (int x = start_x; x < this->box_dimension_x; x += THREADED_GRID_SIZE) {
        for (int y = 0; y < this->box_dimension_y; y++) {
            this->cells[x][y].p = this->cells[x][y].p_next;
        }
    }
}

template <class T>
void ThreadedNavierStokes<T>::enforceVelocityBoundaryConditions(int thread_index) {
    int start_x = thread_index;

    for (int x = start_x; x < this->box_dimension_x; x += THREADED_GRID_SIZE) {
        for (int y = 0; y < this->box_dimension_y; y++) {
            if (this->cells[x][y].u_boundary_set) {
                this->cells[x][y].u = this->cells[x][y].u_boundary;  // enforce the BC if it has been set
                continue;
            }

            // checks if you are on the edge, else do nothing
            if (y == 0) {
                this->cells[x][y].u = this->cells[x][y+1].u; // equal to cell above
            } else if (y == this->box_dimension_y - 1) {
                this->cells[x][y].u = this->cells[x][y-1].u; // equal to cell below
            } else if (x == 0) {
                this->cells[x][y].u = this->cells[x+1][y].u; // equal to cell to right
            } else if (x == this->box_dimension_x - 1) {
                this->cells[x][y].u = this->cells[x-1][y].u;  // equal to cell to left
            }
        }
    }

    for (int x = start_x; x < this->box_dimension_x; x += THREADED_GRID_SIZE) {
        for (int y = 0; y < this->box_dimension_y; y++) {
            if (this->cells[x][y].v_boundary_set) {
                this->cells[x][y].v = this->cells[x][y].v_boundary;  // enforce the BC if it has been set
                continue;
            }

            // checks if you are on the edge, else do nothing
            if (y == 0) {
                this->cells[x][y].v = this->cells[x][y+1].v; // equal to cell above
            } else if (y == this->box_dimension_y - 1) {
                this->cells[x][y].v = this->cells[x][y-1].v; // equal to cell below
            } else if (x == 0) {
                this->cells[x][y].v = this->cells[x+1][y].v; // equal to cell to right
            } else if (x == this->box_dimension_x - 1) {
                this->cells[x][y].v = this->cells[x-1][y].v;  // equal to cell to left
            }
        }
    }
}

template <class T>
void ThreadedNavierStokes<T>::unifiedApproximateTimeStep(int thread_index) {
    int start_x = thread_index + 1;

    for (int x = start_x; x < this->box_dimension_x - 1; x += THREADED_GRID_SIZE) {
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
void ThreadedNavierStokes<T>::unifiedComputeRightHand(int thread_index) {
    int start_x = thread_index + 1;

    for (int x = start_x; x < this->box_dimension_x - 1; x += THREADED_GRID_SIZE) {
        for (int y = 1; y < this->box_dimension_y - 1; y++) {
            // compute the central differences
            this->cells[x][y].du_next_dx = (this->cells[x+1][y].u_next - this->cells[x-1][y].u_next) / 2. / this->element_length_x;
            this->cells[x][y].dv_next_dy = (this->cells[x][y+1].v_next - this->cells[x][y-1].v_next) / 2. / this->element_length_y;

            this->cells[x][y].right_hand_size = (this->density / this->time_step) * (this->cells[x][y].du_next_dx + this->cells[x][y].dv_next_dy);
        }
    }
}

template <class T>
void ThreadedNavierStokes<T>::unifiedVelocityCorrection(int thread_index) {
    int start_x = thread_index + 1;

    for (int x = start_x; x < this->box_dimension_x - 1; x += THREADED_GRID_SIZE) {
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
