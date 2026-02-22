#include <cmath>
#include <cstdlib>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// =-=-=-=-=-= Parameter choice =-=-=-=-=-=-=

// Domain dimension
constexpr unsigned int DIMENSION = 1;

// Domain boundaries
constexpr double X_DOMAIN_LEFT  = 0.0;
constexpr double X_DOMAIN_RIGHT = 12.0;

constexpr double V_DOMAIN_LEFT = -6.0;
constexpr double V_DOMAIN_RIGHT = 6.0;
constexpr unsigned int NV = 1e3; // used only to evaluate rho(x), independent of deal.ii grid

// Global refinement level
constexpr unsigned int GLOBAL_REFINEMENT = 7;

// Polynomial degree
constexpr unsigned int FE_DEGREE = 3;

// f0 parameters
constexpr double EPS = 0.01;
constexpr double WAVE_NR = 0.5;


// =-=-=-=-= f_0(x,v) =-=-=-=-=

double f0(const double x,
          const double v,
          const double eps=EPS,
          const double k=WAVE_NR)
{
  const double prefactor = (1.0 + eps * std::cos(k*x));
  const double gaussian  = (v*v / std::sqrt((2.0 * M_PI)))
                            * std::exp(- 0.5 * v*v);

  return prefactor*gaussian;
}

// =-=-=-=-= Compute rho(x) =-=-=-=-=

double compute_rho(const double x)
{
  const double v_min = V_DOMAIN_LEFT;
  const double v_max = V_DOMAIN_RIGHT;
  const double Nv = NV;

  const double dv = std::abs(v_min - v_max)/Nv;

  double integral = 0.0;

  for (unsigned int i=0; i<Nv; ++i)
  {
    const double v = v_min + (i+0.5)*dv; // Integrate with mid-point rule 
    
    integral += f0(x, v) * dv;
  }

  return 1.0 - integral;
}


// =-=-=-=-= rho(x) in deal.ii =-=-=-=-=

template <int dim>
class ChargeDensity : public Function<dim>
{
public:
  ChargeDensity(double eps, double k)
    : Function<dim>(1), eps(eps), k(k) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override
  {
    return compute_rho(p[0]);
  }

private:
  const double eps;
  const double k;
};

/* ------------------------------
   Poisson Solver
--------------------------------*/
template <int dim>
class PoissonProblem
{
public:
  PoissonProblem(unsigned int degree);
  void run();

private:
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
};


template <int dim>
PoissonProblem<dim>::PoissonProblem(unsigned int degree)
  : fe(degree)
  , dof_handler(triangulation)
{}


template <int dim>
void PoissonProblem<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  // Homogeneous Dirichlet BC
  VectorTools::interpolate_boundary_values(
      dof_handler,
      0,
      Functions::ZeroFunction<dim>(),
      constraints);

  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}


template <int dim>
void PoissonProblem<dim>::assemble_system()
{
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

  ChargeDensity<dim> rhs_function(EPS, WAVE_NR);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs    = 0;

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      const double rho = rhs_function.value(fe_values.quadrature_point(q));

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

  MappingQ1<dim> mapping;
  DoFTools::map_dofs_to_support_points(mapping,
                                       dof_handler,
                                       support_points);

  Vector<double> x_coordinate(dof_handler.n_dofs());

  for (unsigned int i = 0; i < support_points.size(); ++i)
    x_coordinate[i] = support_points[i][0];  // x-component in 1D
                                             

  /* ---- Output density ---- */
  ChargeDensity<dim> rho(EPS, WAVE_NR);
  Vector<double> density(triangulation.n_active_cells());

  DataOut<dim> data_out_rho;
  data_out_rho.attach_dof_handler(dof_handler);

  Vector<double> density_nodal(solution.size());
  VectorTools::interpolate(dof_handler, rho, density_nodal);

  data_out_rho.add_data_vector(density_nodal, "density");
  data_out_rho.add_data_vector(x_coordinate, "x_coordinate");

  data_out_rho.build_patches();

  std::ofstream out1("density.vtk");
  data_out_rho.write_vtk(out1);


  /* ---- Output electric field ---- */
  DataOut<dim> data_out_E;
  data_out_E.attach_dof_handler(dof_handler);

  std::vector<std::string> E_names(dim, "E");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);

  Vector<double> dummy(solution.size() * dim);

  data_out_E.add_data_vector(solution,
                             "potential");

  data_out_E.add_data_vector(solution,
                             "E_field",
                             DataOut<dim>::type_dof_data,
                             interpretation);
  data_out_E.add_data_vector(x_coordinate, "x_coordinate");

  data_out_E.build_patches();

  std::ofstream out2("electric_field.vtk");
  data_out_E.write_vtk(out2);
}


template <int dim>
void PoissonProblem<dim>::run()
{
  GridGenerator::hyper_cube(triangulation,
                            X_DOMAIN_LEFT,
                            X_DOMAIN_RIGHT);

  triangulation.refine_global(GLOBAL_REFINEMENT);

  setup_system();
  assemble_system();
  solve();
  output_results();
}


int main()
{
  try
  {
    PoissonProblem<DIMENSION> poisson_problem(FE_DEGREE);
    poisson_problem.run();
  }
  catch (std::exception &exc)
  {
    std::cerr << exc.what() << std::endl;
    return 1;
  }
  return 0;
}
