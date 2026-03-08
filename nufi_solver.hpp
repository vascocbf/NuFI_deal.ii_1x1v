#ifndef NUFI_SOLVER_HPP
#define NUFI_SOLVER_HPP

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
  double eval_rho(size_t n, double x, double u);

private:

  void compute_density();
  void solve_poisson();
  void update_distribution();

  double evaluate_E(double x);

  PoissonProblem<1> poisson;

  std::vector<double> coeffs;
  std::vector<double> rho;

  unsigned int Nt;
  unsigned int Nx;

  double Lx = Parameters::LX;

  unsigned int order;

  double dt;
  size_t stride_t;
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

inline double NuFISolver::eval_rho(size_t n,
                                      double x,
                                      double u)
{
    if (n == 0)
        return f0(x,u);
   
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

    return compute_rho(x_periodic); // compute_rho (from fields.hpp) uses f0
}

inline void NuFISolver::run()
{
    rho.resize(Nx, 0.0);               // initialize density array

    // Main time-stepping loop
    for (size_t n = 0; n < Nt; ++n)
    {

        // Solve this logic! To use on deal.ii  

        // 1. Compute charge density rho from current distribution
        compute_density();

        // 2. Solve Poisson's equation to update electric field
        solve_poisson();

        // 3. Update distribution function along characteristics
        update_distribution();

        // Optional: compute ftilda at current step if needed
        // for demonstration: evaluate ftilda at midpoint x, u = 0
        // double ft = eval_ftilda(n, 0.5 * poisson.get_Lx(), 0.0);
    }
}

#endif
