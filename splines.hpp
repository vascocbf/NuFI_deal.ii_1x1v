#ifndef SPLINES_HPP
#define SPLINES_HPP

#include <cstddef>

namespace splines1d
{

template <typename real>
constexpr real faculty( size_t n ) noexcept
{
   return  (n > 1) ? real(n)*faculty<real>(n-1) : real(1);
}

template <typename real, size_t order, size_t derivative = 0>
void N( real x, real *result, size_t stride = 1 ) noexcept
{
    static_assert( order > 0, "Splines must have order greater than zero." );
    constexpr int n { order      };
    constexpr int d { derivative };

    if ( derivative >= order )
        for ( size_t i = 0; i < order; ++i )
            result[ i*stride ] = 0;

    if ( n == 1 )
    {
        *result = 1;
        return;
    }

    real v[n]; v[n-1] = 1;
    for ( int k = 1; k < n - d; ++k )
    {
        v[n-k-1] = (1-x)*v[n-k];

        for ( int i = 1-k; i < 0; ++i )
            v[n-1+i] = (x-i)*v[n-1+i] + (k+1+i-x)*v[n+i];

        v[n-1] *= x;
    }

    // Differentiate if necessary.
    for ( size_t j = derivative; j-- > 0;  )
    {
        v[j] = -v[j+1];
        for ( size_t i = j + 1; i < order - 1; ++i )
            v[i] = v[i] - v[i+1];
    }

    constexpr real factor = real(1) / faculty<real>(order-derivative-1);
    for ( size_t i = 0; i < order; ++i )
        result[i*stride] = v[i]*factor;
}

template <typename real, size_t order, size_t derivative = 0>
real eval( real x, const real *coefficients, size_t stride = 1 ) noexcept
{
    static_assert( order > 0, "Splines must have order greater than zero." );
    static_assert( order > derivative, "Too high derivative requested." );
    constexpr size_t n { order };
    constexpr size_t d { derivative };

    if ( d >= n ) return 0;
    if ( n == 1 ) return *coefficients;

    // Gather coefficients.
    real c[ order ];
    for ( size_t j = 0; j < order; ++j )
        c[j] = coefficients[ stride * j ];

    // Differentiate if necessary.
    for ( size_t j = 1; j <= d; ++j )
        for ( size_t i = n; i-- > j; )
            c[i] = c[i] - c[i-1];

    // Evaluate using de Boor’s algorithm.
    for ( size_t j = 1; j < n-d; ++j )
        for ( size_t i = n-d; i-- > j; )
            c[d+i] = (x+n-d-1-i)*c[d+i] + (i-j+1-x)*c[d+i-1];

    constexpr real factor = real(1) / faculty<real>(order-derivative-1);
    return factor*c[n-1];
}

}

#endif


