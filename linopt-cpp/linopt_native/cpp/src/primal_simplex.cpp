#include "linopt_native/primal_simplex.hpp"

#include "linopt_native/simplex_utils.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

namespace linopt_native {

PrimalSimplex::PrimalSimplex(PivotRule pivot_rule, LogCallback log_callback)
    : pivot_strategy_(make_primal_pivoting_strategy(pivot_rule)),
      log_callback_(std::move(log_callback)) {}

SimplexResult PrimalSimplex::solve(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c,
    const Eigen::VectorXi& initial_basis,
    int max_iterations) {
  validate_basis(a, initial_basis);

  SimplexResult result;
  result.profile.enabled = native_profile_enabled();
  Eigen::VectorXi basis = initial_basis;
  auto non_basic_vars = non_basic_variables(static_cast<int>(c.size()), basis);
  Eigen::MatrixXd inv_basis = columns(a, basis).inverse();
  Eigen::VectorXd x_basis = inv_basis * b;
  {
    ProfileScope profile_scope(&result.profile, "pivot_initialize");
    pivot_strategy_->initialize(a, basis, non_basic_vars, inv_basis);
  }
  double objective = elements(c, basis).dot(x_basis);
  append_history(result, basis, objective);
  if (log_callback_) {
    log_callback_(initial_objective_message(objective));
    log_callback_("Iter     Objective      Primal Inf.    Dual Inf.    Time");
  }
  const auto start = std::chrono::steady_clock::now();

  for (int iteration = 1; iteration < max_iterations; ++iteration) {
    Eigen::VectorXd rc;
    {
      ProfileScope profile_scope(&result.profile, "reduced_costs");
      rc = reduced_costs(a, c, basis, non_basic_vars, inv_basis);
    }
    if ((rc.array() >= -kNonNegativityTolerance).all()) {
      result.iterations = iteration - 1;
      result.basis = basis;
      result.solution = Eigen::VectorXd::Zero(c.size());
      for (int i = 0; i < basis.size(); ++i) {
        result.solution[basis[i]] = x_basis[i];
      }
      result.objective = result.objective_history.back();
      if (log_callback_) {
        log_callback_(optimal_message(result.objective, iteration - 1));
      }
      return result;
    }

    int entering_index = -1;
    {
      ProfileScope profile_scope(&result.profile, "pick_entering");
      entering_index = pivot_strategy_->pick_entering_index(rc, non_basic_vars);
    }
    const int entering_variable = non_basic_vars[static_cast<std::size_t>(entering_index)];
    Eigen::VectorXd d;
    {
      ProfileScope profile_scope(&result.profile, "basic_direction");
      d = inv_basis * a.col(entering_variable);
    }
    if ((d.array() <= kPivotTolerance).all()) {
      result.status = "unbounded";
      result.message = "Primal simplex detected an unbounded problem.";
      return result;
    }
    int exiting_index = -1;
    {
      ProfileScope profile_scope(&result.profile, "pick_exiting");
      exiting_index = pivot_strategy_->pick_exiting_index(basis, x_basis, d, inv_basis);
    }
    const double x_entering = x_basis[exiting_index] / d[exiting_index];

    non_basic_vars[static_cast<std::size_t>(entering_index)] = basis[exiting_index];
    basis[exiting_index] = entering_variable;
    const bool recompute_inverse = iteration % kInverseRecomputeInterval == 0;
    {
      ProfileScope profile_scope(&result.profile, "inverse_update");
      inv_basis = recompute_inverse
                      ? columns(a, basis).inverse()
                      : update_inverse(a, inv_basis, entering_variable, exiting_index);
    }
    x_basis -= x_entering * d;
    x_basis[exiting_index] = x_entering;
    if (recompute_inverse) {
      ProfileScope profile_scope(&result.profile, "pivot_initialize");
      pivot_strategy_->initialize(a, basis, non_basic_vars, inv_basis);
    }
    objective = elements(c, basis).dot(x_basis);
    append_history(result, basis, objective);
    if (log_callback_ && (iteration < 10 || iteration % 100 == 0)) {
      ProfileScope profile_scope(&result.profile, "progress_log");
      const double primal_inf = (columns(a, basis) * x_basis - b).array().abs().sum();
      const Eigen::VectorXd dual_values = a.transpose() * (inv_basis * elements(c, basis)) - c;
      const double dual_inf = std::max(0.0, dual_values.sum());
      log_callback_(simplex_progress_row(
          iteration, objective, primal_inf, dual_inf, elapsed_seconds(start)));
    }
  }

  result.status = "iteration_limit";
  result.message = "Primal simplex reached the iteration limit.";
  if (log_callback_) {
    log_callback_(iteration_limit_message(max_iterations));
  }
  result.iterations = std::max(0, max_iterations - 1);
  result.basis = basis;
  result.solution = Eigen::VectorXd::Zero(c.size());
  for (int i = 0; i < basis.size(); ++i) {
    result.solution[basis[i]] = x_basis[i];
  }
  result.objective = result.objective_history.empty() ? 0.0 : result.objective_history.back();
  return result;
}

SimplexResult solve_primal_simplex_dense(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c,
    const Eigen::VectorXi& initial_basis,
    int max_iterations,
    PivotRule pivot_rule,
    LogCallback log_callback) {
  PrimalSimplex solver(pivot_rule, std::move(log_callback));
  return solver.solve(a, b, c, initial_basis, max_iterations);
}

}  // namespace linopt_native
