#include "nufi/nufi_solver.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/fe_field_function.h>

#include <iostream>
#include <memory>
#include <ostream>
#include <cstddef>
#include <vector>

#include "nufi/parameters.h"
#include "nufi/save_results.h"
#include "nufi/poisson_problem.h"
#include "nufi/fields.h"

using namespace dealii;

double NuFISolver::eval_ftilda(unsigned int n,
                                      double x,
                                      double u,
                                      const double *E_coeffs) const
{
  if ( n == 0 ) return f0(x,u);

  const size_t stride_x = 1;
  const size_t stride_t = stride_x*(Nx + Parameters::SPLINE_ORDER - 1);

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

double NuFISolver::eval_rho(const unsigned int n,
                          const double x,
                          const double *E_coeffs,
                          const unsigned int Nv) const
{
  const double dv = (Parameters::V_DOMAIN_RIGHT - Parameters::V_DOMAIN_LEFT) / Nv;
  const double v_min = Parameters::V_DOMAIN_LEFT;

  double integral = 0.0;
  for (unsigned int i = 0; i < Nv; ++i)
    integral += eval_ftilda(n, x, v_min + i * dv, E_coeffs) * dv;
  
  return 1.0 - integral;
}

void NuFISolver::run()
{
  std::cout << "Building E_sline\n\n";

  using std::abs;
  using std::max;

  const size_t stride_t = Nx + order - 1;

  std::unique_ptr<double[]> coeffs { new double[ Nt*stride_t ] {} };
  std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,sizeof(double)*Nx)), std::free };

  if ( rho == nullptr ) throw std::bad_alloc {};

  Gradient grad(x_min, x_max, Nx);

  for (unsigned int it = 0; it < Nt; ++it)
  {
      std::cout << "Timestep " << it << " / " << Nt << std::endl << std::endl;

      // compute rho

      double dx = Parameters::SPLINE_DX;
      double x = Parameters::X_DOMAIN_LEFT;

    	for(size_t i = 0; i<Nx; i++, x+=dx)
    	{
        double ith_rho = eval_rho(it, x, coeffs.get(),Parameters::NV); 
        AssertThrow(std::isfinite(ith_rho), ExcMessage("NaN detected in rho"));
    		rho.get()[i] =  ith_rho;
    	}

      poisson.set_rhs_function(std::make_unique<ChargeDensity_NuFI<1>>(rho.get(), Nx));
      poisson.solve_step();

      std::vector<double> sampled_potential = poisson.sample_electric_potential(x_min, x_max, Nx);

      save_space_vector(sampled_potential, "potential", it);

      std::vector<double> E_vals = grad.compute(sampled_potential); 
      // std::vector<double> E_vals = poisson.sample_electric_field(x_min, x_max, Nx);
      
      save_space_vector(E_vals, "electric", it);

      double* current_coeffs = coeffs.get() + it*stride_t;
      interpolate<double, Parameters::SPLINE_ORDER>(current_coeffs, E_vals.data());

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

NuFISolver::NuFISolver()
  : order(Parameters::FE_DEGREE),
    poisson(order)
{
  std::cout << "Initializing dealii Poisson Solver\n";
  poisson.initialize();
}
