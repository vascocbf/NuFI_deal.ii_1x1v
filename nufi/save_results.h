#ifndef SAVE_RESULTS_H
#define SAVE_RESULTS_H

#include <string>
#include "nufi/nufi_solver.h"


void save_f( const NuFISolver &solver,
    unsigned int n,
                                    const double *E_coeffs,
                                    unsigned int Nx_out,
                                    unsigned int Nv_out,
                                    const std::string &filename);

void save_rho(const NuFISolver &solver,
    unsigned int n,
                                    const double *E_coeffs,
                                    unsigned int Nx_out,
                                    const std::string &filename);

void save_Efield(unsigned int n,
                                    const double *E_coeffs,
                                    unsigned int Nx_out,
                                    const std::string &filename);

void save_space_vector(const std::vector<double>& vals, const std::string& filename, size_t it);

#endif 
