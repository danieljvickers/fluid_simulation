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

template<class T>
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

template<class T>
int NavierStokesSolver<T>::getCellIndex(int const x_index, int const y_index) {
    if (x_index >= box_dimension_x || y_index >= box_dimension_y) {
        return -1;
    }
    return y_index * box_dimension_y + x_index;
}

template<class T>
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

template<class T>
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

template<class T>
void NavierStokesSolver<T>::computeTimeDerivitive(int const index_x, int const index_y) {
    //  get the indeices of neightboring cells
    int index = getCellIndex(index_x, index_y);

    // get the time derivitives
    cells[index].du_dt = kinematic_viscosity * cells[index].u_laplacian - cells[index].u * cells[index].du_dx - cells[index].v * cells[index].du_dy;
    cells[index].du_dt = kinematic_viscosity * cells[index].u_laplacian - cells[index].u * cells[index].du_dx - cells[index].v * cells[index].du_dy;
}

template<class T>
void NavierStokesSolver<T>::takeTimeStep(int const index_x, int const index_y) {
    //  get the indeices of neightboring cells
    int index = getCellIndex(index_x, index_y);

    // get the time derivitives
    cells[index].u_next = cells[index].u + time_step * cells[index].du_dt;
    cells[index].v_next = cells[index].v + time_step * cells[index].dv_dt;
}

template<class T>
void NavierStokesSolver<T>::computeRightHandSide(int const index_x, int const index_y) {
    int index = getCellIndex(index_x, index_y);
    cells[index].right_hand_size = (density / time_step) * (cells[index].du_dx + cells[index].dv_dy);
}
