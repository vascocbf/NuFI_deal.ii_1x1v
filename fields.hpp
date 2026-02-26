#ifndef FIELDS_HPP
#define FIELDS_HPP

#include <cmath>
#include <deal.II/base/function.h>
#include "parameters.hpp"

using namespace dealii;

inline double f0(const double x,
                 const double v,
                 const double eps = Parameters::EPS,
                 const double k   = Parameters::WAVE_NR)
{
  const double prefactor = (1.0 + eps * std::cos(k*x));
  const double gaussian  =
      (v*v / std::sqrt(2.0 * M_PI)) *
      std::exp(-0.5 * v*v);

  return prefactor * gaussian;
}

inline double compute_rho(const double x,
                          const unsigned int Nv = Parameters::NV)
{
  const double dv = (Parameters::V_DOMAIN_RIGHT - Parameters::V_DOMAIN_LEFT) / Nv;

  double integral = 0.0;

  for (unsigned int i = 0; i < Nv; ++i)
  {
    const double v = Parameters::V_DOMAIN_LEFT + (i + 0.5) * dv;
    integral += f0(x, v) * dv;
  }

  return 1.0 - integral;
}

template <int dim>
class ChargeDensity : public Function<dim>
{
public:
  ChargeDensity(double eps,
                double k,
                unsigned int Nv)
    : Function<dim>(1), eps(eps), k(k), Nv(Nv) {}

  virtual double value(const Point<dim> &p,
                       [[maybe_unused]] const unsigned int component = 0) const override
  {
    return compute_rho(p[0], Nv);
  }

private:
  const double eps;
  const double k;
  const unsigned int Nv;
};


#endif
