/*
Todo: 
- update Nx between timesteps to account for adaptivity changes because Vector rho needs to be resized
 */

#ifndef NUFI_SOLVER_HPP
#define NUFI_SOLVER_HPP

#include <cmath>
#include <cstdlib>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/fe_field_function.h>

#include <vector>
#include <cstddef>

#include "parameters.hpp"
#include "poisson_problem.hpp"
#include "fields.hpp" // holds f0(x,v), and compute_rho(x)

using namespace dealii;

class NuFISolver
{
public:
  NuFISolver();

  void run();
  double eval_rho(unsigned int n, double x, unsigned int Nv = Parameters::NV);

private:

  double eval_ftilda(unsigned int n, double x, double u);
  void solve_poisson(unsigned int n);

  double evaluate_E(double x);
  
  std::vector<double> rho;

  unsigned int Nt = std::floor(Parameters::TMAX/Parameters::DT);
  unsigned int Nx;

  double Lx = Parameters::LX;

  unsigned int order;

  double dt = Parameters::DT;

  PoissonProblem<1> poisson;

};

inline double NuFISolver::evaluate_E(double x)
{
  // Wrap x into the periodic domain
  double x_periodic = x - Lx * std::floor(x / Lx);
  Point<1> p(x_periodic);

  Functions::FEFieldFunction<1> E_field(
        poisson.get_dof_handler(),
        poisson.get_solution()
    );

  double E_val = 0.0;

  try
  {
      // Evaluate the electric field at point p
      // If your solution represents phi, take negative gradient
      Tensor<1,1> grad = E_field.gradient(p);
      E_val = -grad[0];  // -∂φ/∂x
  }
  catch (const VectorTools::ExcPointNotAvailableHere &)
  {
      // This happens if p lies in an artificial cell in parallel
      AssertThrow(false, ExcMessage("Point not available on this process."));
  }

  return E_val;
}

inline double NuFISolver::eval_ftilda(unsigned int n,
                                      double x,
                                      double u)
{
  double Lu = std::abs(Parameters::V_DOMAIN_LEFT - Parameters::V_DOMAIN_RIGHT);

  if (n == 0)
      return f0(x, u);
 
  double Ex;

  // Initial half-step.
  Ex = evaluate_E(x);
  u += 0.5*dt*Ex;

  while ( --n )
  {
      x -= dt*u;
      Ex = evaluate_E(x);
      u += dt*Ex;
  }

  // Final half-step.
  x -= dt*u;
  Ex = evaluate_E(x);
  u += 0.5*dt*Ex; // is this line useless ?
  
  double x_periodic = x - Lx * std::floor(x / Lx);
  double u_periodic = u - Lu * std::floor(u / Lu);

  return f0(x_periodic, u_periodic);
}

inline double NuFISolver::eval_rho(const unsigned int n,
                          const double x,
                          const unsigned int Nv)
{
  const double dv = (Parameters::V_DOMAIN_RIGHT - Parameters::V_DOMAIN_LEFT) / Nv;

  double integral = 0.0;

  for (unsigned int i = 0; i < Nv; ++i)
  {
    const double v = Parameters::V_DOMAIN_LEFT + (i + 0.5) * dv;
    integral += eval_ftilda(n, x, v) * dv;
  }

  return 1.0 - integral;
}

class ChargeDensity_NuFI : public Function<1>
{
public:
  ChargeDensity_NuFI(NuFISolver &solver, size_t n)
      : solver(solver), n(n) {}

  virtual double value(const Point<1> &p,
                       [[maybe_unused]] const unsigned int component = 0) const override
  {
      double x = p[0];

      return solver.eval_rho(n, x); 
      // u not used anymore
  }

private:
  NuFISolver &solver;
  size_t n;
};


inline void NuFISolver::solve_poisson(unsigned int n)
{
  ChargeDensity_NuFI rho_function(*this, n);

  poisson.set_rhs_function(rho_function);

  poisson.solve_step();
}

inline void NuFISolver::run()
{
  std::cout << "Starting NuFI solver\n";

  for (unsigned int n = 0; n < Nt; ++n)
  {
    std::cout << "Timestep " << n << " / " << Nt << std::endl;


    double dx = Lx / Nx;

    for (unsigned int i = 0; i < Nx; ++i)
    {
      double x = (i + 0.5) * dx;
      rho[i] = eval_rho(n, x);
    }

    solve_poisson(n);
  }

  std::cout << "NuFI simulation finished.\n";
}

inline NuFISolver::NuFISolver()
  : order(Parameters::FE_DEGREE),
    poisson(order, Parameters::NV)
{
  poisson.initialize();

  Nx = poisson.get_dof_handler().n_dofs();
  rho.resize(Nx, 0.0);

}
#endif
