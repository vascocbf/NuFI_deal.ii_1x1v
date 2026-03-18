#ifndef NUFI_SOLVER_H
#define NUFI_SOLVER_H

#include <vector>
#include <cmath>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/fe_field_function.h>

#include "nufi/parameters.h"
#include "nufi/poisson_problem.h"
#include "nufi/fields.h"

using namespace dealii;

class NuFISolver
{
public:
  NuFISolver();

  void run();
  double eval_rho(unsigned int n, double x, const double *E_coeffs, unsigned int Nv = Parameters::NV) const;
  double eval_ftilda(unsigned int n, double x, double u, const double *E_coeffs) const;

private:


  unsigned int Nt = std::floor(Parameters::TMAX/Parameters::DT);
  unsigned int Nx = Parameters::SPLINE_NX;

  double Lx = Parameters::LX;

  std::vector<double> rho;

  unsigned int order;

  PoissonProblem<1> poisson;

};

template<unsigned int dim>
class ChargeDensity_NuFI : public Function<dim>
{
  public:
    ChargeDensity_NuFI(const double *rho_values, unsigned int Nx)
      : Function<dim>(), rho(rho_values), Nx(Nx) {}

    virtual double value(const Point<dim> &p,
                         [[maybe_unused]] const unsigned int component = 0) const override
    {
      const double x = p[0];

      // Map x -> grid index
      const double L = Parameters::LX;
      const double dx = L / (Nx-1);

      int i = static_cast<int>(std::floor((x - Parameters::X_DOMAIN_LEFT) / dx));

      // periodic wrap
      i = (i % Nx + Nx) % Nx;

      return rho[i];
    }

  private:
    const double *rho;
    const unsigned int Nx;
};

#endif
