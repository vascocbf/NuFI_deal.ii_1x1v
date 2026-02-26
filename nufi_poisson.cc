#include <iostream>

#include "parameters.hpp"
#include "poisson_problem.hpp"

int main()
{
  try
  {
    PoissonProblem<Parameters::DIMENSION> poisson_problem(Parameters::FE_DEGREE,
                                                          Parameters::NV);

    poisson_problem.run();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << "Exception: "
              << std::endl
              << exc.what()
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << "Unknown exception!"
              << std::endl;
    return 1;
  }

  return 0;
}

