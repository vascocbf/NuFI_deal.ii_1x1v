#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <cmath>
#include <cstdlib>
namespace Parameters
{
  constexpr unsigned int DIMENSION = 1;

  constexpr double X_DOMAIN_LEFT  = 0.0;
  constexpr double X_DOMAIN_RIGHT = 4*M_PI;
  constexpr double LX = std::abs(X_DOMAIN_RIGHT- X_DOMAIN_LEFT);

  constexpr double V_DOMAIN_LEFT = -10.0;
  constexpr double V_DOMAIN_RIGHT = 10.0;

  constexpr unsigned int NV = 512;

  constexpr unsigned int GLOBAL_REFINEMENT = 8;
  constexpr unsigned int FE_DEGREE = 4;

  constexpr double EPS = 0.01;
  constexpr double WAVE_NR = 0.5;
  constexpr double F0_FACTOR = 0.39894228040143267793994;

  // NUFI options
  constexpr double DT=1./16.;
  constexpr unsigned int TMAX = 10;

  //spline options
  constexpr int SPLINE_NX = 512;
  constexpr double SPLINE_DX = LX/SPLINE_NX;

  //Plotting options
  constexpr int PLOT_FREQUENCY = 2;
}

#endif
