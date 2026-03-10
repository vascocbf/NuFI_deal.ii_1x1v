#include <iostream>
#include <filesystem>
#include "nufi_solver.hpp"

int main()
{
  try
  {
    // Delete all files in results/ directory
        std::string results_dir = "results";
        if (std::filesystem::exists(results_dir) && std::filesystem::is_directory(results_dir))
        {
            for (const auto& entry : std::filesystem::directory_iterator(results_dir))
            {
                if (std::filesystem::is_regular_file(entry.path()))
                {
                  std::filesystem::remove(entry.path());
                    std::cout << "Deleted: " << entry.path() << "\n";
                }
            }
        }

    //Used in the past to test the basic poisson problem
    //
    // PoissonProblem<Parameters::DIMENSION> poisson_problem(Parameters::FE_DEGREE,
    //                                                       Parameters::NV);
    //
    // poisson_problem.run();
    
    NuFISolver solver;
    solver.run();
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

