//
// Created by dan on 12/10/24.
//

#ifndef PARRALLELNAVIERSTOKES_CUH
#define PARRALLELNAVIERSTOKES_CUH

#include "../cpp/NavierStokesCell.h"
#include "../cpp/NavierStokesSolver.h"


tamplate <class T>
class ParrallelNavierStokes : public NavierStokesSolver {
private:
    NavierStokesCell<T>* d_cells;

public:
    ParrallelNavierStokes(int box_dim_x, int box_dim_y, T domain_size_x, T domain_size_y);
    ~ParrallelNavierStokes();

    void solve();
};

// explicit instantiation allows float and double precision types
template class NavierStokesSolver<float>;
template class NavierStokesSolver<double>;

#endif //PARRALLELNAVIERSTOKES_CUH
