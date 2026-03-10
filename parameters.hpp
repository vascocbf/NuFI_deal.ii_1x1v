#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <cstdlib>
namespace Parameters
{
  constexpr unsigned int DIMENSION = 1;

  constexpr double X_DOMAIN_LEFT  = 0.0;
  constexpr double X_DOMAIN_RIGHT = 12.0;
  constexpr double LX = std::abs(X_DOMAIN_RIGHT- X_DOMAIN_LEFT);

  constexpr double V_DOMAIN_LEFT = -6.0;
  constexpr double V_DOMAIN_RIGHT = 6.0;

  constexpr unsigned int NV = 1000;

  constexpr unsigned int GLOBAL_REFINEMENT = 7;
  constexpr unsigned int FE_DEGREE = 3;

  constexpr double EPS = 0.01;
  constexpr double WAVE_NR = 0.5;

  // NUFI options
  constexpr double DT=0.05;
  constexpr unsigned int TMAX = 2;

  //spline options
  constexpr int SPLINE_NX = 256;
}

#endif
