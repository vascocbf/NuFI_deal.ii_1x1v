#ifndef FIELDS_H
#define FIELDS_H

#include <cmath>
#include <deal.II/base/function.h>
#include "nufi/parameters.h"
#include "nufi/splines.h"
#include "nufi/lsmr.h"

using namespace dealii;

inline double f0(const double x,
                 const double v,
                 const double eps = Parameters::EPS,
                 const double k   = Parameters::WAVE_NR)
{
  const double prefactor = Parameters::F0_FACTOR * (1.0 + eps * std::cos(k*x));
  const double gaussian  = v*v  * std::exp(-0.5 * v*v);

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


template <size_t dx = 0>
double eval(double x, const double *coeffs) noexcept
{
    using std::floor;

    // Shift to a box that starts at 0.
    x -= Parameters::X_DOMAIN_LEFT;

    // Get "periodic position" in box at origin.
    x = x - Parameters::LX * floor( x/Parameters::LX ); 

    // Knot number
    double x_knot = floor( x/Parameters::SPLINE_DX); 

    size_t ii = static_cast<size_t>(x_knot);

    // Convert x to reference coordinates.
    x = x/Parameters::SPLINE_DX - x_knot;

    // Scale according to derivative.
    double factor = 1;
    for ( size_t i = 0; i < dx; ++i ) factor *= 1/Parameters::SPLINE_DX;

    return factor*splines1d::eval<double,Parameters::SPLINE_ORDER,dx>(x, coeffs + ii);
}

template <typename real, size_t order>
void interpolate( real *coeffs, const real *values)
{
    std::unique_ptr<real[]> tmp { new real[ Parameters::SPLINE_NX ] };

    for ( size_t i = 0; i < Parameters::SPLINE_NX; ++i )
        tmp[ i ] = coeffs[ i ];

    struct mat_t
    {
        real  N[ order ];

        mat_t()
        {
            splines1d::N<real,order>(0,N);
        }

        void operator()( const real *in, real *out ) const
        {
            for ( size_t i = 0; i < Parameters::SPLINE_NX; ++i )
            {
                real result = 0;
                if ( i + order <= Parameters::SPLINE_NX )
                {
                    for ( size_t ii = 0; ii < order; ++ii )
                        result += N[ii] * in[ i + ii ];
                }
                else
                {
                    for ( size_t ii = 0; ii < order; ++ii )
                        result += N[ii]*in[ (i+ii) % Parameters::SPLINE_NX];
                }
                out[ i ] = result;
            }
        }
    };

    struct transposed_mat_t
    {
        real  N[ order ];

        transposed_mat_t()
        {
            splines1d::N<real,order>(0,N);
        }

        void operator()( const real *in, real *out ) const
        {
            for ( size_t i = 0; i < Parameters::SPLINE_NX; ++i )
                out[ i ] = 0;

            for ( size_t i = 0; i < Parameters::SPLINE_NX; ++i )
            {
                if ( i + order <= Parameters::SPLINE_NX )
                {
                    for ( size_t ii = 0; ii < order; ++ii )
                        out[ i + ii ]  += N[ii] * in[ i ];
                }
                else
                {
                    for ( size_t ii = 0; ii < order; ++ii )
                        out[ (i+ii) % Parameters::SPLINE_NX ] += N[ii]*in[ i ];
                }
            }
        }
    };

    mat_t M; transposed_mat_t Mt;
    lsmr_options<real> opt; opt.silent = true;
    lsmr( Parameters::SPLINE_NX, Parameters::SPLINE_NX , M, Mt, values, tmp.get(), opt );

    if ( opt.iter == opt.max_iter )
        std::cerr << "Warning. LSMR did not converge.\n";

    for ( size_t i = 0; i < Parameters::SPLINE_NX + order - 1; ++i )
        coeffs[ i ] = tmp[ i % Parameters::SPLINE_NX ];
}

class Gradient {
public:
    Gradient(double xmin, double xmax, unsigned int Nx)
        : xmin_(xmin), xmax_(xmax), Nx_(Nx)
    {
        if (xmax_ <= xmin_) {
            throw std::invalid_argument("xmax must be greater than xmin");
        }
    }

    std::vector<double> compute(const std::vector<double>& values) const {
        size_t n = values.size();
        if (n < 2) {
            throw std::invalid_argument("Need at least 2 points");
        }

        std::vector<double> grad(n);

        double dx = (xmax_ - xmin_) / (n-1);
        // periodic boundaries
        grad[0]     = -(values[1]   - values[n-1]) / (2.0 * dx);
        grad[n-1]   = -(values[0]   - values[n-2]) / (2.0 * dx);

        for (size_t i = 1; i < n-1; ++i) {
            grad[i] = -(values[i+1] - values[i-1]) / (2.0 * dx);
        }


        return grad;
    }

private:
    double xmin_;
    double xmax_;
    [[maybe_unused]] unsigned int Nx_;
};

// Not used, only outputs f0
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
