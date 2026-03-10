#ifndef SPLINE_FIELD_HPP
#define SPLINE_FIELD_HPP

#include <iostream>
#include <stdexcept>
#include <cstddef>
#include <vector>
#include <cmath>

template<typename Real, size_t Order>
class UniformSpline1D
{
public:

    struct Config
    {
        size_t Nx;
        Real x_min;
        Real Lx;

        Real dx;
        Real dx_inv;
        Real Lx_inv;
    };

    Config config;
    std::vector<Real> coeffs;

public:

    UniformSpline1D(const std::vector<Real>& values,
                    Real x_min,
                    Real x_max)
    {
        config.Nx = values.size();
        config.x_min = x_min;
        config.Lx = x_max - x_min;

        config.dx = config.Lx / (config.Nx-1);
        config.dx_inv = 1.0 / config.dx;
        config.Lx_inv = 1.0 / config.Lx;

        coeffs.resize(config.Nx + Order - 1);

        interpolate(values);
    }

    Real eval(Real x) const
    {
      x -= config.x_min;
      x = std::fmod(x, config.Lx);
      if (x<0) x+= config.Lx;

      Real x_cell = x * config.dx_inv;
      size_t i = std::min(static_cast<size_t>(std::floor(x_cell)), config.Nx - 1);

      Real local_x = x_cell - i;

      Real basis[Order];
      basisFunctions(local_x, basis);

      Real result = 0;

      for(size_t j = 0; j < Order; ++j)
      {
        size_t idx = (i+j) % coeffs.size();
        result += basis[j] * coeffs[idx];
      }

      return result;
    }

private:

    static void basisFunctions(Real x, Real* N)
    {
        Real v[Order];
        v[Order-1] = 1;

        for(size_t k = 1; k < Order; ++k)
        {
            v[Order-k-1] = (1-x)*v[Order-k];

            for(int i = 1-k; i < 0; ++i)
                v[Order-1+i] = (x-i)*v[Order-1+i] + (k+1+i-x)*v[Order+i];

            v[Order-1] *= x;
        }

        Real factor = 1;
        for(size_t i=2;i<Order;i++) factor*=i;

        for(size_t i=0;i<Order;i++)
            N[i] = v[i]/factor;
    }

    void interpolate(const std::vector<Real>& values)
    {

      std::cout << "interpolating";
        size_t N = config.Nx;

        std::vector<Real> rhs(values);
        std::vector<std::vector<Real>> A(N, std::vector<Real>(N,0));

        Real Nbasis[Order];
        basisFunctions(0, Nbasis);

        for(size_t i=0;i<N;i++)
        {
            for(size_t j=0;j<Order;j++)
            {
                size_t col = (i+j)%N;
                A[i][col] = Nbasis[j];
            }
        }

        // naive Gaussian elimination
        std::vector<Real> x = rhs;

        for(size_t k=0;k<N;k++)
        {
            Real pivot = A[k][k];
            for(size_t j=k;j<N;j++)
                A[k][j] /= pivot;

            x[k] /= pivot;

            for(size_t i=k+1;i<N;i++)
            {
                Real f = A[i][k];

                for(size_t j=k;j<N;j++)
                    A[i][j] -= f*A[k][j];

                x[i] -= f*x[k];
            }
        }

        for(int i=N-1;i>=0;i--)
        {
            for(size_t j=i+1;j<N;j++)
                x[i] -= A[i][j]*x[j];
        }

        for(size_t i=0;i<N;i++)
            coeffs[i] = x[i];

        for(size_t i=0;i<Order-1;i++)
            coeffs[N+i] = coeffs[i];
    }
};

#endif // !SPLINE_FIELD_HPP
