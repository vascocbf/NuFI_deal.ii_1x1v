#ifndef POISSON_NON_PERIODIC_HPP
#define POISSON_NON_PERIODIC_HPP

#include "parameters.hpp"
#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
 
#include <deal.II/fe/fe_q.h>
 
#include <deal.II/dofs/dof_tools.h>
 
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
 
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
 
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
 
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
 
using namespace dealii;
 
 
template<int dim> 
class Poisson_non_periodic
{
public:
  Poisson_non_periodic ();
 
  void run();
 
  void initialize();
  void solve_step();

  void set_rhs_function(std::unique_ptr<Function<dim>> rhs_function);

  const Vector<double> &get_solution() const { return solution; }
  const DoFHandler<dim> &get_dof_handler() const { return dof_handler; }

  std::vector<double> sample_electric_field(const Poisson_non_periodic<dim> &problem,  // sampling to save as spline
                                            unsigned int Nx,
                                            double x_min,
                                            double x_max);
  void output_results(unsigned int n);
private:

  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;
 
  Triangulation<1> triangulation;
  const FE_Q<1>    fe;
  DoFHandler<1>    dof_handler;
 
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
 
  Vector<double> solution;
  Vector<double> system_rhs;

  std::unique_ptr<const Function<dim>> rhs_function;

};
 
template<int dim> 
Poisson_non_periodic<dim>::Poisson_non_periodic()
  : fe(/* polynomial degree = */ 1)
  , dof_handler(triangulation)
{}

template <int dim>
void Poisson_non_periodic<dim>::set_rhs_function(std::unique_ptr<Function<dim>> rhs)
{
  rhs_function = std::move(rhs);
}
 
 
template<int dim>
void Poisson_non_periodic<dim>::make_grid()
{
  Point<dim, double> x0 = Parameters::X_DOMAIN_RIGHT;
  Point<dim, double> x1 = Parameters::X_DOMAIN_RIGHT;
  GridGenerator::hyper_rectangle(triangulation, x0, x1);
  triangulation.refine_global(Parameters::GLOBAL_REFINEMENT);
 
  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}
 
 
 
template<int dim>
void Poisson_non_periodic<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
 
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
 
  system_matrix.reinit(sparsity_pattern);
 
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}
 
 
template<int dim>
void Poisson_non_periodic<dim>::assemble_system()
{
  const QGauss<1> quadrature_formula(fe.degree + 1);
  FEValues<1> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values);
 
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
 
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
 
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
 
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
 
      cell_matrix = 0;
      cell_rhs    = 0;
 
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          
          const double rho = rhs_function->value(fe_values.quadrature_point(q_index));

          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx
 
          for (const unsigned int i : fe_values.dof_indices())
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            rho *                                // f(x_q)
                            fe_values.JxW(q_index));            // dx
        }
      cell->get_dof_indices(local_dof_indices);
 
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));
 
      for (const unsigned int i : fe_values.dof_indices())
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
 
 
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           types::boundary_id(0),
                                           Functions::ZeroFunction<1>(),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}
 
 
template<int dim> 
void Poisson_non_periodic<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-6 * system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
 
  std::cout << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
}
 
template <int dim>
void Poisson_non_periodic<dim>::output_results(unsigned int n)
{

  // --- extract DoF coordinates ---
  std::vector<Point<dim>> support_points(dof_handler.n_dofs());
  Vector<double> x_coordinate(dof_handler.n_dofs());

  for (unsigned int i = 0; i < support_points.size(); ++i)
    x_coordinate[i] = support_points[i][0];  // x-component in 1D
                                             
  //---- Output density ----
  ChargeDensity<dim> rho(Parameters::EPS, Parameters::WAVE_NR, Parameters::NV);

  DataOut<dim> data_out_rho;
  data_out_rho.attach_dof_handler(dof_handler);

  Vector<double> density(solution.size());
  VectorTools::interpolate(dof_handler, rho, density);

  data_out_rho.add_data_vector(density, "density");
  data_out_rho.add_data_vector(x_coordinate, "x_coordinate");

  data_out_rho.build_patches();

  std::ofstream out1("results/density_" + std::to_string(n) + ".vtk");
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

  std::ofstream out2("results/electric_field_"+ std::to_string(n)+".vtk");
  data_out_E.write_vtk(out2);
}

template <int dim>
void Poisson_non_periodic<dim>::initialize()
{
  make_mesh();      // build grid
  setup_system();     // distribute DoFs and matrices
}

template <int dim>
void Poisson_non_periodic<dim>::solve_step()
{
  assemble_system();
  solve();
}

#endif
