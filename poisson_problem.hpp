#ifndef POISSON_PROBLEM_HPP
#define POISSON_PROBLEM_HPP

#include <deal.II/base/function.h>
#include <fstream>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "parameters.hpp"
#include "fields.hpp"

using namespace dealii;

// =-=-=-=-= Poisson Solver =-=-=-=-=

template <int dim>
class PoissonProblem
{
public:
  PoissonProblem(unsigned int degree, unsigned int Nv);

  void initialize();
  void solve_step();
  void run();

  void set_Nv(unsigned int new_Nv);
  void set_rhs_function(const Function<dim> &rhs);

  const Vector<double> &get_solution() const { return solution; }
  const DoFHandler<dim> &get_dof_handler() const { return dof_handler; }

private:
  void create_mesh();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;      // phi
  Vector<double> system_rhs;

  const Function<dim> *rhs_function;

  MappingQ<dim> mapping;

  unsigned int Nv;
};


template <int dim>
void PoissonProblem<dim>::set_Nv(unsigned int new_Nv)
{
  Nv = new_Nv;
}

template <int dim>
void PoissonProblem<dim>::set_rhs_function(const Function<dim> &rhs)
{
  rhs_function = &rhs;
}

template <int dim>
PoissonProblem<dim>::PoissonProblem(unsigned int degree, unsigned int Nv)
  : fe(degree)
  , dof_handler(triangulation)
  , mapping(degree)
  , Nv(Nv)
{}


// =-=-=-=-= Make Grid =-=-=-=-=

template<int dim>
void PoissonProblem<dim>::create_mesh()
{

  std::cout << "Creating Mesh\n";
  GridGenerator::hyper_cube(triangulation,
                            Parameters::X_DOMAIN_LEFT,
                            Parameters::X_DOMAIN_RIGHT);

  // Make x-dim boundaries periodic
  Tensor<1, dim> offset;
  std::vector<GridTools::PeriodicFacePair<
              typename Triangulation<dim>::cell_iterator>> periodicity_vector;

  GridTools::collect_periodic_faces(triangulation,
                                    0,
                                    1,
                                    0,
                                    periodicity_vector,
                                    offset);
  
  triangulation.add_periodicity(periodicity_vector);

  triangulation.refine_global(Parameters::GLOBAL_REFINEMENT);
}

template <int dim>
void PoissonProblem<dim>::setup_system()
{

  std::cout << "Setting up Poisson system\n";
  dof_handler.distribute_dofs(fe);

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  // 'boundary' condition phi(x_0) = 0 
  constraints.add_line(0);
  constraints.set_inhomogeneity(0, 0.0);

  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

// =-=-=-=-= E_field = -dPhi/dx =-=-=-=-=

template <int dim>
class ElectricFieldPostprocessor : public DataPostprocessorVector<dim>
{
public:
  ElectricFieldPostprocessor()
    : DataPostprocessorVector<dim>("electric_field", update_gradients)
  {}

  virtual void evaluate_scalar_field(
    const DataPostprocessorInputs::Scalar<dim> &input_data,
    std::vector<Vector<double>> &computed_quantities) const override
  {
    AssertDimension(input_data.solution_gradients.size(),
                    computed_quantities.size());

    for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        AssertDimension(computed_quantities[p].size(), dim);
        for (unsigned int d = 0; d < dim; ++d)
          computed_quantities[p][d] = -input_data.solution_gradients[p][d];
      }
  }
};


// =-=-=-=-= Poisson equation solver =-=-=-=-=

template <int dim>
void PoissonProblem<dim>::assemble_system()
{
  std::cout << "Assembling Poisson System\n";
  QGauss<dim>  quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values |
                          update_gradients |
                          update_quadrature_points |
                          update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  Assert(rhs_function != nullptr, ExcMessage("RHS function not set"));

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs    = 0;

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      const double rho = rhs_function->value(fe_values.quadrature_point(q));

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          cell_matrix(i, j) +=
              fe_values.shape_grad(i, q) *
              fe_values.shape_grad(j, q) *
              fe_values.JxW(q);

        cell_rhs(i) +=
            fe_values.shape_value(i, q) *
            rho *
            fe_values.JxW(q);
      }
    }

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(cell_matrix,
                                           cell_rhs,
                                           local_dof_indices,
                                           system_matrix,
                                           system_rhs);
  }
}


template <int dim>
void PoissonProblem<dim>::solve()
{

  std::cout << "Calling PoissonProblem::solve()\n";
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);

  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  constraints.distribute(solution);
}


template <int dim>
void PoissonProblem<dim>::output_results() const
{

  // --- extract DoF coordinates ---
  std::vector<Point<dim>> support_points(dof_handler.n_dofs());

  DoFTools::map_dofs_to_support_points(mapping,
                                       dof_handler,
                                       support_points);

  Vector<double> x_coordinate(dof_handler.n_dofs());

  for (unsigned int i = 0; i < support_points.size(); ++i)
    x_coordinate[i] = support_points[i][0];  // x-component in 1D
                                             
  //---- Output density ----
  ChargeDensity<dim> rho(Parameters::EPS, Parameters::WAVE_NR, Nv);

  DataOut<dim> data_out_rho;
  data_out_rho.attach_dof_handler(dof_handler);

  Vector<double> density(solution.size());
  VectorTools::interpolate(dof_handler, rho, density);

  data_out_rho.add_data_vector(density, "density");
  data_out_rho.add_data_vector(x_coordinate, "x_coordinate");

  data_out_rho.build_patches();

  std::ofstream out1("density.vtk");
  data_out_rho.write_vtk(out1);

  //---- Output electric field & potential ----
  DataOut<dim> data_out_E;
  data_out_E.attach_dof_handler(dof_handler);

  ElectricFieldPostprocessor<dim> electric_field;
  Vector<double> dummy(solution.size() * dim);

  data_out_E.add_data_vector(solution, "potential");
  data_out_E.add_data_vector(solution, electric_field);
  data_out_E.add_data_vector(x_coordinate, "x_coordinate");

  data_out_E.build_patches();

  std::ofstream out2("electric_field.vtk");
  data_out_E.write_vtk(out2);
}

template <int dim>
void PoissonProblem<dim>::initialize()
{
  set_Nv(Parameters::NV);

  create_mesh();      // build grid
  setup_system();     // distribute DoFs and matrices
}

template <int dim>
void PoissonProblem<dim>::solve_step()
{
  system_matrix = 0;
  system_rhs = 0;
  std::cout << "Calling PoissonProblem::solve_step()\n";
  assemble_system();
  solve();
}


// NuFI doesnt use this, kept only for testing. 
template <int dim>
void PoissonProblem<dim>::run()
{
  set_Nv(Parameters::NV); // dont use anywhere else! Other functions still use Parameters::NV.
  create_mesh();
  setup_system();
  assemble_system();
  solve();
  output_results();
}

#endif
