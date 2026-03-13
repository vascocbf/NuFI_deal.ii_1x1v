#ifndef NUFI_BLAS_HPP
#define NUFI_BLAS_HPP

#include <cstddef>

/*!
 * \brief Convenience wrappers for BLAS, with overloads for single and double
 *        precision.
 */
namespace blas
{

double dot( const size_t n, const double *x, size_t incx,
                            const double *y, size_t incy );

float  dot( const size_t n, const float  *x, size_t incx,
                            const float  *y, size_t incy );

void axpy( size_t n, double alpha, const double *x, size_t incx,
                                         double *y, size_t incy );

void axpy( size_t n, float  alpha, const float  *x, size_t incx,
                                         float  *y, size_t incy );


void scal( size_t n, double alpha, double *x, size_t incx );
void scal( size_t n, float  alpha, float  *x, size_t incx );

void copy( size_t n, const double *x, size_t incx, double *y, size_t incy );
void copy( size_t n, const float  *x, size_t incx, float  *y, size_t incy );

void ger( const size_t M, const size_t N, const double alpha,
          const double *X, const size_t incX, const double *Y, const size_t incY,
          double *A, const size_t lda);

void ger( const size_t M, const size_t N, const float alpha,
          const float *X, const size_t incX, const float *Y, const size_t incY,
          float *A, const size_t lda);


void gemv( const char trans, size_t m, size_t n,
		   double alpha, const double *a, size_t lda,
           const double *x, size_t incx, double beta,
           double *y, size_t incy );

void gemv( const char trans, size_t m, size_t n,
		   float alpha, const float *a, size_t lda,
           const float *x, size_t incx, float beta,
           float *y, size_t incy );
}


#endif

