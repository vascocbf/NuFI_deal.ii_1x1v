#ifndef NUFI_SOLVER_HPP
#define NUFI_SOLVER_HPP

#include <cmath>
#include <cstdlib>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/fe_field_function.h>

#include <iostream>
#include <vector>
#include <cstddef>

#include "parameters.hpp"
#include "poisson_problem.hpp"
#include "fields.hpp" // holds f0(x,v), and compute_rho(x)
#include "spline_field.hpp"

using namespace dealii;

class NuFISolver
{
public:
  NuFISolver();

  void run();
  double eval_rho(unsigned int n, double x, const UniformSpline1D<double,4>& E_spline, unsigned int Nv = Parameters::NV);

private:

  double eval_ftilda(unsigned int n, double x, double u, const UniformSpline1D<double, 4>& E_spline);

  std::vector<double> rho;

  unsigned int Nt = std::floor(Parameters::TMAX/Parameters::DT);
  unsigned int Nx = Parameters::SPLINE_NX;

  double Lx = Parameters::LX;

  unsigned int order;

  double dt = Parameters::DT;

  PoissonProblem<1> poisson;

};

inline double NuFISolver::eval_ftilda(unsigned int n,
                                      double x,
                                      double u,
                                      const UniformSpline1D<double, 4>& E_spline)
{
  double Lu = std::abs(Parameters::V_DOMAIN_LEFT - Parameters::V_DOMAIN_RIGHT);

  if (n == 0)
      return f0(x, u);

  // Initial half-step.
  u += 0.5*dt*E_spline.eval(x);

  while ( --n )
  {
      x -= dt*u;
      u += dt*E_spline.eval(x);
  }

  // Final half-step.
  x -= dt*u;
  u += 0.5*dt*E_spline.eval(x);

  double x_periodic = x - Lx * std::floor(x / Lx);
  double u_periodic = u - Lu * std::floor(u / Lu);

  return f0(x_periodic, u_periodic);
}

inline double NuFISolver::eval_rho(const unsigned int n,
                          const double x,
                          const UniformSpline1D<double, 4>& E_spline,
                          const unsigned int Nv)
{
  const double dv = (Parameters::V_DOMAIN_RIGHT - Parameters::V_DOMAIN_LEFT) / Nv;

  double integral = 0.0;
  for (unsigned int i = 0; i < Nv; ++i)
  {
    const double v = Parameters::V_DOMAIN_LEFT + (i + 0.5) * dv;
    AssertThrow(std::isfinite(E_spline.eval(x)), ExcMessage("NaN detected in E_spline.eval(x) inside NuFISolver::eval_rho integral loop"));
    integral += eval_ftilda(n, x, v, E_spline) * dv;
  }
  
  return 1.0 - integral;
}

class ChargeDensity_NuFI : public Function<1>
{
public:
  ChargeDensity_NuFI(NuFISolver &solver, size_t n, const UniformSpline1D<double,4> &E_spline)
      : solver(solver), n(n), E_spline(E_spline) {}

  virtual double value(const Point<1> &p,
                       [[maybe_unused]] const unsigned int component = 0) const override
  {
      double x = p[0];

      return solver.eval_rho(n, x, E_spline); 
  }

private:
  NuFISolver &solver;
  size_t n;
  const UniformSpline1D<double, 4> &E_spline;
};


inline void NuFISolver::run()
{
  std::cout << "Start of NuFISolver::run()\n";
  
  // init E_spline

  unsigned int Nx = Parameters::SPLINE_NX;
  // Nx grid points
  double dx = Lx / (Nx-1);

  std::vector<double> E_grid(Nx);

  //set initial E points
  for (unsigned int i=0; i<Nx; ++i)
  {
    [[maybe_unused]] double x = Parameters::X_DOMAIN_LEFT + i*dx;
    E_grid[i] = 1; 
  }

  UniformSpline1D<double,4> E_spline(E_grid, Parameters::X_DOMAIN_LEFT, Parameters::X_DOMAIN_RIGHT);

  for (unsigned int it = 0; it < Nt; ++it)
  {
      std::cout << "Timestep " << it << " / " << Nt << std::endl;

      // Step 1: Evaluate rho^n(x) using current E_spline
      std::cout << "Start of eval_rho loop\n";
      std::vector<double> rho(Nx);
      for (unsigned int i = 0; i < (Nx); ++i)
      {
          double x = (i + 0.5) * dx;
          rho[i] = eval_rho(it, x, E_spline, Parameters::NV);
      }
      std::cout << "End of eval_rho loop\n";

      ChargeDensity_NuFI rho_function(*this, it, E_spline);

      poisson.set_rhs_function(rho_function);

      for (unsigned int i=0; i< rho.size(); ++i) // check for bad rho[i]
      {
        AssertThrow(std::isfinite(rho[i]), ExcMessage("NaN detected in rho"));
      }

      poisson.solve_step();


      E_grid = poisson.sample_electric_field(poisson, Nx, 0.0, Lx);

      // Step 4: Build spline for E^{n+1} (used in next time step)
      E_spline = UniformSpline1D<double, 4>(E_grid, 0.0, Lx);
  }

  std::cout << "NuFI simulation finished.\n";
}

inline NuFISolver::NuFISolver()
  : order(Parameters::FE_DEGREE),
    poisson(order)
{
  std::cout << "Initializing Poisson\n";
  poisson.initialize();

  Nx = poisson.get_dof_handler().n_dofs();
  rho.resize(Nx, 0.0);

}
#endif
