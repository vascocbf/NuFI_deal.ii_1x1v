#ifndef NUFI_SOLVER_HPP
#define NUFI_SOLVER_HPP

#include <cmath>
#include <cstdlib>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/fe_field_function.h>

#include <iostream>
#include <memory>
#include <ostream>
#include <vector>
#include <cstddef>

#include "parameters.hpp"
#include "save_results.hpp"
#include "poisson_problem.hpp"
#include "fields.hpp"

using namespace dealii;

class NuFISolver
{
public:
  NuFISolver();

  void run();
  double eval_rho(unsigned int n, double x, const double *E_coeffs, unsigned int Nv = Parameters::NV);
  double eval_ftilda(unsigned int n, double x, double u, const double *E_coeffs);

  // void save_ftilda(unsigned int n, const double *E_coeffs, unsigned int Nx_out, unsigned int Nv_out, const std::string &filename);
  // void save_rho(unsigned int n, const double *E_coeffs, unsigned int Nx_out, const std::string &filename);
  // void save_Efield(unsigned int n, const double *E_coeffs, unsigned int Nx_out, const std::string &filename);

private:


  unsigned int Nt = std::floor(Parameters::TMAX/Parameters::DT);
  unsigned int Nx = Parameters::SPLINE_NX;

  double Lx = Parameters::LX;

  std::vector<double> rho;

  unsigned int order;

  PoissonProblem<1> poisson;

};

inline double NuFISolver::eval_ftilda(unsigned int n,
                                      double x,
                                      double u,
                                      const double *E_coeffs)

{
  if ( n == 0 ) return f0(x,u);

  const size_t stride_x = 1;
  const size_t stride_t = stride_x*(Parameters::SPLINE_NX + Parameters::SPLINE_ORDER - 1);

  double Ex;
  const double *c;

  // We omit the initial half-step.

  while ( --n )
  {
      x  = x - Parameters::DT *u;
      c  = E_coeffs + n*stride_t;
      Ex = -eval<1>(x, c);
      u  = u + Parameters::DT *Ex;
  }

  // The final half-step.
  x -= Parameters::DT*u;
  c  = E_coeffs + n*stride_t;
  Ex = -eval<1>(x, c);
  u += 0.5*Parameters::DT*Ex;

  return f0(x,u);
}

inline double NuFISolver::eval_rho(const unsigned int n,
                          const double x,
                          const double *E_coeffs,
                          const unsigned int Nv)
{
  const double dv = (Parameters::V_DOMAIN_RIGHT - Parameters::V_DOMAIN_LEFT) / Nv;
  const double v_min = Parameters::V_DOMAIN_LEFT;

  double integral = 0.0;
  for (unsigned int i = 0; i < Nv; ++i)
    integral += eval_ftilda(n, x, v_min + i * dv, E_coeffs) * dv;
  
  return 1.0 - integral;
}

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




inline void NuFISolver::run()
{
  std::cout << "Building E_sline\n\n";

  using std::abs;
  using std::max;

  const size_t stride_t = Nx + order - 1;

  std::unique_ptr<double[]> coeffs { new double[ Nt*stride_t ] {} };
  std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,sizeof(double)*Nx)), std::free };

  if ( rho == nullptr ) throw std::bad_alloc {};

  for (unsigned int it = 0; it < Nt; ++it)
  {
      std::cout << "Timestep " << it << " / " << Nt << std::endl << std::endl;

      // compute rho

      double dx = Parameters::SPLINE_DX;
      double x = Parameters::X_DOMAIN_LEFT;

    	for(size_t i = 0; i<Nx; i++, x+=dx)
    	{
        double ith_rho = eval_rho(it, x, coeffs.get(), Parameters::NV); 
        AssertThrow(std::isfinite(ith_rho), ExcMessage("NaN detected in rho"));
    		rho.get()[i] =  ith_rho;
    	}

      // bool rho_written = false;
      //
      // if (!rho_written)
      // {
      //     std::ofstream rho_file("rho_initial.dat");
      //
      //     if (!rho_file)
      //         throw std::runtime_error("Could not open rho_initial.dat");
      //
      //     rho_file.precision(16);
      //     rho_file << std::scientific;
      //
      //     for (size_t i = 0; i < Nx; ++i)
      //         rho_file << rho.get()[i] << "\n";
      //
      //     rho_written = true;
      // }

      poisson.set_rhs_function(std::make_unique<ChargeDensity_NuFI<1>>(rho.get(), Parameters::SPLINE_NX));
      poisson.solve_step();

      if (it % Parameters::PLOT_FREQUENCY == 0)
      {
        std::cout << "Saving results... \n\n";
        save_ftilda(*this, it, coeffs.get(), 128, 128, "results/ftilda_" + std::to_string(it) + ".dat");
        save_rho(*this, it, coeffs.get(), 128, "results/rho_" + std::to_string(it) + ".dat");
        save_Efield(it, coeffs.get(), 128, "results/field_" + std::to_string(it) + ".dat");
      }
  }

  std::cout << "NuFI simulation finished.\n";
}

inline NuFISolver::NuFISolver()
  : order(Parameters::FE_DEGREE),
    poisson(order)
{
  std::cout << "Initializing dealii Poisson Solver\n";
  poisson.initialize();
}

#endif
