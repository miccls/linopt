#include "linopt_native/solvers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <set>

namespace linopt_native {
namespace {

Eigen::MatrixXd columns(const Eigen::MatrixXd& a, const Eigen::VectorXi& indices) {
  Eigen::MatrixXd out(a.rows(), indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    out.col(i) = a.col(indices[i]);
  }
  return out;
}

Eigen::MatrixXd columns(const Eigen::MatrixXd& a, const std::vector<int>& indices) {
  Eigen::MatrixXd out(a.rows(), static_cast<int>(indices.size()));
  for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
    out.col(i) = a.col(indices[static_cast<std::size_t>(i)]);
  }
  return out;
}

Eigen::VectorXd elements(const Eigen::VectorXd& values, const Eigen::VectorXi& indices) {
  Eigen::VectorXd out(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    out[i] = values[indices[i]];
  }
  return out;
}

Eigen::VectorXd elements(const Eigen::VectorXd& values, const std::vector<int>& indices) {
  Eigen::VectorXd out(static_cast<int>(indices.size()));
  for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
    out[i] = values[indices[static_cast<std::size_t>(i)]];
  }
  return out;
}

Eigen::VectorXd reduced_costs(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& c,
    const Eigen::VectorXi& basis,
    const std::vector<int>& non_basic_vars,
    const Eigen::MatrixXd& inv_basis) {
  const Eigen::VectorXd lambda = inv_basis.transpose() * elements(c, basis);
  return elements(c, non_basic_vars) - columns(a, non_basic_vars).transpose() * lambda;
}

void append_history(SimplexResult& result, const Eigen::VectorXi& basis, double objective) {
  result.basis_history.push_back(basis);
  result.objective_history.push_back(objective);
}

void validate_basis(const Eigen::MatrixXd& a, const Eigen::VectorXi& basis) {
  if (basis.size() != a.rows()) {
    throw LinoptNativeError("Initial basis length must equal the number of constraints.");
  }
  std::set<int> seen;
  for (int i = 0; i < basis.size(); ++i) {
    if (basis[i] < 0 || basis[i] >= a.cols()) {
      throw LinoptNativeError("Initial basis contains an out-of-range variable index.");
    }
    if (!seen.insert(basis[i]).second) {
      throw LinoptNativeError("Initial basis contains duplicate variable indices.");
    }
  }
}

Eigen::VectorXd steepest_weights_for_basis(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXi& basis,
    const std::vector<int>& non_basic_vars,
    const Eigen::MatrixXd& inv_basis) {
  const Eigen::MatrixXd directions = inv_basis * columns(a, non_basic_vars);
  return Eigen::VectorXd::Ones(directions.cols()) +
         directions.array().square().colwise().sum().transpose().matrix();
}

std::string simplex_progress_row(
    int iteration,
    double objective,
    double primal_inf,
    double dual_inf,
    double elapsed_seconds) {
  std::ostringstream out;
  out << std::setw(4) << iteration << "    " << std::scientific << std::setprecision(3)
      << std::setw(10) << objective << "     " << std::setw(10) << primal_inf
      << "     " << std::setw(10) << dual_inf << "    " << std::defaultfloat
      << std::setprecision(4) << elapsed_seconds << "s";
  return out.str();
}

std::string initial_objective_message(double objective) {
  std::ostringstream out;
  out << "Initial objective value " << objective;
  return out.str();
}

std::string optimal_message(double objective, int iterations) {
  std::ostringstream out;
  out << "Simplex algorithm found optimal objective " << objective << " after "
      << iterations << " iterations.";
  return out.str();
}

std::string iteration_limit_message(int max_iterations) {
  std::ostringstream out;
  out << "Simplex algorithm terminated due to " << max_iterations << " iteration limit";
  return out.str();
}

double elapsed_seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

std::vector<int> non_basic_variables(int num_variables, const Eigen::VectorXi& basis) {
  std::set<int> basic;
  for (int i = 0; i < basis.size(); ++i) {
    basic.insert(basis[i]);
  }
  std::vector<int> out;
  out.reserve(static_cast<std::size_t>(num_variables - basis.size()));
  for (int variable = 0; variable < num_variables; ++variable) {
    if (!basic.contains(variable)) {
      out.push_back(variable);
    }
  }
  return out;
}

int primal_entering_index(
    const Eigen::VectorXd& reduced_costs,
    const std::vector<int>& non_basic_vars,
    PivotRule pivot_rule,
    const Eigen::VectorXd* steepest_weights) {
  int best_index = -1;
  double best_score = std::numeric_limits<double>::infinity();
  int best_variable = std::numeric_limits<int>::max();
  for (int i = 0; i < reduced_costs.size(); ++i) {
    if (reduced_costs[i] >= -kPivotTolerance) {
      continue;
    }
    double score = reduced_costs[i];
    if (pivot_rule == PivotRule::SteepestEdge) {
      if (steepest_weights == nullptr || steepest_weights->size() != reduced_costs.size()) {
        throw LinoptNativeError("Steepest-edge weights are not aligned with reduced costs.");
      }
      score = reduced_costs[i] / std::sqrt(std::max((*steepest_weights)[i], kPivotTolerance));
    }
    const int variable = non_basic_vars[static_cast<std::size_t>(i)];
    if (pivot_rule == PivotRule::Bland) {
      if (variable < best_variable) {
        best_variable = variable;
        best_index = i;
      }
    } else if (score < best_score) {
      best_score = score;
      best_index = i;
    }
  }
  if (best_index < 0) {
    throw LinoptNativeError("No valid entering variable found.");
  }
  return best_index;
}

int primal_exiting_index(
    const Eigen::VectorXi& basis,
    const Eigen::VectorXd& x_basis,
    const Eigen::VectorXd& basic_direction) {
  int best_index = -1;
  double best_ratio = std::numeric_limits<double>::infinity();
  int best_variable = std::numeric_limits<int>::max();
  for (int i = 0; i < basic_direction.size(); ++i) {
    if (basic_direction[i] <= kPivotTolerance) {
      continue;
    }
    const double ratio = std::max(0.0, x_basis[i]) / basic_direction[i];
    if (ratio < best_ratio - 1e-14 ||
        (std::abs(ratio - best_ratio) <= 1e-14 && basis[i] < best_variable)) {
      best_ratio = ratio;
      best_variable = basis[i];
      best_index = i;
    }
  }
  if (best_index < 0) {
    throw LinoptNativeError("No valid exiting variable found.");
  }
  return best_index;
}

int dual_exiting_index(
    const Eigen::VectorXd& x_basis,
    const Eigen::VectorXi& basis,
    const Eigen::MatrixXd& inv_basis,
    PivotRule pivot_rule) {
  int best_index = -1;
  if (pivot_rule == PivotRule::DualBland) {
    int best_variable = std::numeric_limits<int>::max();
    for (int i = 0; i < x_basis.size(); ++i) {
      if (x_basis[i] < -kPivotTolerance && basis[i] < best_variable) {
        best_variable = basis[i];
        best_index = i;
      }
    }
  } else if (pivot_rule == PivotRule::DualDantzig) {
    x_basis.minCoeff(&best_index);
  } else if (pivot_rule == PivotRule::DualSteepestEdge) {
    double best_score = std::numeric_limits<double>::infinity();
    for (int i = 0; i < x_basis.size(); ++i) {
      if (x_basis[i] >= -kPivotTolerance) {
        continue;
      }
      const double norm = std::sqrt(std::max(inv_basis.row(i).squaredNorm(), kPivotTolerance));
      const double score = x_basis[i] / norm;
      if (score < best_score) {
        best_score = score;
        best_index = i;
      }
    }
  } else {
    throw LinoptNativeError("Dual simplex requires a dual pivot rule.");
  }
  if (best_index < 0 || x_basis[best_index] >= -kPivotTolerance) {
    throw LinoptNativeError("No valid exiting variable found.");
  }
  return best_index;
}

int dual_entering_index(
    const std::vector<int>& non_basic_vars,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& pivot_direction) {
  Eigen::VectorXi vars(static_cast<int>(non_basic_vars.size()));
  for (int i = 0; i < vars.size(); ++i) {
    vars[i] = non_basic_vars[static_cast<std::size_t>(i)];
  }
  return primal_exiting_index(vars, s, pivot_direction);
}

Eigen::MatrixXd update_inverse(
    const Eigen::MatrixXd& a,
    const Eigen::MatrixXd& b_inv,
    int entering_variable,
    int exiting_index) {
  Eigen::MatrixXd updated = b_inv;
  const Eigen::VectorXd eta = b_inv * a.col(entering_variable);
  const double pivot = eta[exiting_index];
  if (std::abs(pivot) <= kPivotTolerance) {
    throw LinoptNativeError("Pivot is too close to zero.");
  }
  const Eigen::RowVectorXd pivot_row = b_inv.row(exiting_index) / pivot;
  for (int row = 0; row < b_inv.rows(); ++row) {
    if (row == exiting_index) {
      updated.row(row) = pivot_row;
    } else {
      updated.row(row) = b_inv.row(row) - eta[row] * pivot_row;
    }
  }
  return updated;
}

SimplexResult solve_primal_simplex_dense(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c,
    const Eigen::VectorXi& initial_basis,
    int max_iterations,
    PivotRule pivot_rule,
    LogCallback log_callback) {
  validate_basis(a, initial_basis);
  if (pivot_rule != PivotRule::Bland && pivot_rule != PivotRule::Dantzig &&
      pivot_rule != PivotRule::SteepestEdge) {
    throw LinoptNativeError("Primal simplex requires a primal pivot rule.");
  }

  SimplexResult result;
  Eigen::VectorXi basis = initial_basis;
  auto non_basic_vars = non_basic_variables(static_cast<int>(c.size()), basis);
  Eigen::MatrixXd inv_basis = columns(a, basis).inverse();
  Eigen::VectorXd x_basis = inv_basis * b;
  Eigen::VectorXd steepest_weights;
  if (pivot_rule == PivotRule::SteepestEdge) {
    steepest_weights = steepest_weights_for_basis(a, basis, non_basic_vars, inv_basis);
  }
  double objective = elements(c, basis).dot(x_basis);
  append_history(result, basis, objective);
  if (log_callback) {
    log_callback(initial_objective_message(objective));
    log_callback("Iter     Objective      Primal Inf.    Dual Inf.    Time");
  }
  const auto start = std::chrono::steady_clock::now();

  for (int iteration = 1; iteration < max_iterations; ++iteration) {
    const Eigen::VectorXd rc = reduced_costs(a, c, basis, non_basic_vars, inv_basis);
    if ((rc.array() >= -kNonNegativityTolerance).all()) {
      result.iterations = iteration - 1;
      result.basis = basis;
      result.solution = Eigen::VectorXd::Zero(c.size());
      for (int i = 0; i < basis.size(); ++i) {
        result.solution[basis[i]] = x_basis[i];
      }
      result.objective = result.objective_history.back();
      if (log_callback) {
        log_callback(optimal_message(result.objective, iteration - 1));
      }
      return result;
    }

    const int entering_index = primal_entering_index(
        rc, non_basic_vars, pivot_rule,
        pivot_rule == PivotRule::SteepestEdge ? &steepest_weights : nullptr);
    const int entering_variable = non_basic_vars[static_cast<std::size_t>(entering_index)];
    const Eigen::VectorXd d = inv_basis * a.col(entering_variable);
    if ((d.array() <= kPivotTolerance).all()) {
      result.status = "unbounded";
      result.message = "Primal simplex detected an unbounded problem.";
      return result;
    }
    const int exiting_index = primal_exiting_index(basis, x_basis, d);
    const double x_entering = x_basis[exiting_index] / d[exiting_index];

    non_basic_vars[static_cast<std::size_t>(entering_index)] = basis[exiting_index];
    basis[exiting_index] = entering_variable;
    inv_basis = iteration % kInverseRecomputeInterval == 0
                    ? columns(a, basis).inverse()
                    : update_inverse(a, inv_basis, entering_variable, exiting_index);
    x_basis -= x_entering * d;
    x_basis[exiting_index] = x_entering;
    if (pivot_rule == PivotRule::SteepestEdge) {
      steepest_weights = steepest_weights_for_basis(a, basis, non_basic_vars, inv_basis);
    }
    objective = elements(c, basis).dot(x_basis);
    append_history(result, basis, objective);
    if (log_callback && (iteration < 10 || iteration % 100 == 0)) {
      const double primal_inf = (columns(a, basis) * x_basis - b).array().abs().sum();
      const Eigen::VectorXd dual_values = a.transpose() * (inv_basis * elements(c, basis)) - c;
      const double dual_inf = std::max(0.0, dual_values.sum());
      log_callback(simplex_progress_row(
          iteration, objective, primal_inf, dual_inf, elapsed_seconds(start)));
    }
  }

  result.status = "iteration_limit";
  result.message = "Primal simplex reached the iteration limit.";
  if (log_callback) {
    log_callback(iteration_limit_message(max_iterations));
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

SimplexResult solve_dual_simplex_dense(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c,
    const Eigen::VectorXi& initial_basis,
    int max_iterations,
    PivotRule pivot_rule,
    LogCallback log_callback) {
  validate_basis(a, initial_basis);
  SimplexResult result;
  Eigen::VectorXi basis = initial_basis;
  auto non_basic_vars = non_basic_variables(static_cast<int>(c.size()), basis);
  Eigen::MatrixXd inv_basis = columns(a, basis).inverse();
  Eigen::VectorXd x_basis = inv_basis * b;
  double objective = elements(c, basis).dot(x_basis);
  append_history(result, basis, objective);
  if (log_callback) {
    log_callback(initial_objective_message(objective));
    log_callback("Iter     Objective      Primal Inf.    Dual Inf.    Time");
  }
  const auto start = std::chrono::steady_clock::now();

  for (int iteration = 1; iteration < max_iterations; ++iteration) {
    if ((x_basis.array() >= -kNonNegativityTolerance).all()) {
      result.iterations = iteration - 1;
      result.basis = basis;
      result.solution = Eigen::VectorXd::Zero(c.size());
      for (int i = 0; i < basis.size(); ++i) {
        result.solution[basis[i]] = x_basis[i];
      }
      result.objective = result.objective_history.back();
      if (log_callback) {
        log_callback(optimal_message(result.objective, iteration - 1));
      }
      return result;
    }

    const int exiting_index = dual_exiting_index(x_basis, basis, inv_basis, pivot_rule);
    const Eigen::VectorXd s_non_basic = reduced_costs(a, c, basis, non_basic_vars, inv_basis);
    const Eigen::VectorXd v = -inv_basis.row(exiting_index).transpose();
    const Eigen::VectorXd non_basic_direction = columns(a, non_basic_vars).transpose() * v;
    if (non_basic_direction.maxCoeff() <= kPivotTolerance) {
      result.status = "infeasible";
      result.message = "Dual simplex detected primal infeasibility.";
      return result;
    }

    const int entering_index = dual_entering_index(non_basic_vars, s_non_basic, non_basic_direction);
    const int entering_variable = non_basic_vars[static_cast<std::size_t>(entering_index)];
    const Eigen::VectorXd basic_direction = inv_basis * a.col(entering_variable);
    const double gamma = x_basis[exiting_index] / basic_direction[exiting_index];
    x_basis -= gamma * basic_direction;
    x_basis[exiting_index] = gamma;

    non_basic_vars[static_cast<std::size_t>(entering_index)] = basis[exiting_index];
    basis[exiting_index] = entering_variable;
    inv_basis = iteration % kInverseRecomputeInterval == 0
                    ? columns(a, basis).inverse()
                    : update_inverse(a, inv_basis, entering_variable, exiting_index);
    objective = elements(c, basis).dot(x_basis);
    append_history(result, basis, objective);
    if (log_callback && (iteration < 10 || iteration % 100 == 0)) {
      const double primal_inf =
          (columns(a, basis) * x_basis - b).array().abs().sum() -
          x_basis.array().min(0.0).sum();
      const double dual_inf = std::abs(std::min(s_non_basic.minCoeff(), 0.0));
      log_callback(simplex_progress_row(
          iteration, objective, primal_inf, dual_inf, elapsed_seconds(start)));
    }
  }

  result.status = "iteration_limit";
  result.message = "Dual simplex reached the iteration limit.";
  if (log_callback) {
    log_callback(iteration_limit_message(max_iterations));
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

}  // namespace linopt_native
