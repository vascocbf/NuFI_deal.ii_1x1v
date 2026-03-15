#ifndef SAVE_RESULTS_HPP
#define SAVE_RESULTS_HPP

#include "parameters.hpp"
#include <fstream>
#include <string>
#include "nufi_solver.hpp"


inline void save_ftilda( const NuFISolver &solver,
    unsigned int n,
                                    const double *E_coeffs,
                                    unsigned int Nx_out,
                                    unsigned int Nv_out,
                                    const std::string &filename)
{
  std::ofstream file(filename);

  double xmin = Parameters::X_DOMAIN_LEFT;
  double xmax = Parameters::X_DOMAIN_RIGHT;

  double vmin = Parameters::V_DOMAIN_LEFT;
  double vmax = Parameters::V_DOMAIN_RIGHT;

  double dx = (xmax - xmin) / Nx_out;
  double dv = (vmax - vmin) / Nv_out;

  file << Nx_out << " " << Nv_out << "\n";
  file << xmin << " " << xmax << "\n";
  file << vmin << " " << vmax << "\n";

  for (unsigned int i = 0; i < Nx_out; ++i)
  {
      double x = xmin + (i + 0.5)*dx;

      for (unsigned int j = 0; j < Nv_out; ++j)
      {
          double v = vmin + (j + 0.5)*dv;

          double val = solver.eval_ftilda(n, x, v, E_coeffs);

          file << val;

          if (j < Nv_out - 1)
              file << " ";
      }

      file << "\n";
  }

  file.close();
}

inline void save_rho(const NuFISolver &solver,
    unsigned int n,
                                    const double *E_coeffs,
                                    unsigned int Nx_out,
                                    const std::string &filename)
{
  std::ofstream file(filename);

  double xmin = Parameters::X_DOMAIN_LEFT;
  double xmax = Parameters::X_DOMAIN_RIGHT;
  double dx = (xmax - xmin) / Nx_out;

  file << Nx_out << "\n";
  file << xmin << " " << xmax << "\n";

  for (unsigned int i = 0; i < Nx_out; ++i, xmin += dx)
  {
      double val = solver.eval_rho(n, xmin, E_coeffs);
      file << val;
      file << "\n";
  }
  file.close();
}

inline void save_Efield(unsigned int n,
                                    const double *E_coeffs,
                                    unsigned int Nx_out,
                                    const std::string &filename)
{
  std::ofstream file(filename);

  double xmin = Parameters::X_DOMAIN_LEFT;
  double xmax = Parameters::X_DOMAIN_RIGHT;
  double dx = (xmax - xmin) / Nx_out;

  // select from E_coeffs
  const size_t stride_x = 1;
  const size_t stride_t = stride_x*(Parameters::SPLINE_NX + Parameters::SPLINE_ORDER - 1);
  const double *c;
  c  = E_coeffs + n*stride_t;

  file << Nx_out << "\n";
  file << xmin << " " << xmax << "\n";

  for (unsigned int i = 0; i < Nx_out; ++i, xmin += dx)
  {
      double val = -eval<1>(xmin, c);
      file << val;
      file << "\n";
  }
  file.close();
}

#endif 
