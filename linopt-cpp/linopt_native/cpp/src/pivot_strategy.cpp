#include "linopt_native/pivot_strategy.hpp"

#include "linopt_native/simplex_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace linopt_native {
namespace {

int index_of_smallest_ratio(
    const Eigen::VectorXi& variables,
    const Eigen::VectorXd& numerators,
    const Eigen::VectorXd& denominators) {
  int best_index = -1;
  double best_ratio = std::numeric_limits<double>::infinity();
  int best_variable = std::numeric_limits<int>::max();
  for (int i = 0; i < denominators.size(); ++i) {
    if (denominators[i] <= kPivotTolerance) {
      continue;
    }
    const double ratio = std::max(0.0, numerators[i]) / denominators[i];
    if (ratio < best_ratio - 1e-14 ||
        (std::abs(ratio - best_ratio) <= 1e-14 && variables[i] < best_variable)) {
      best_ratio = ratio;
      best_variable = variables[i];
      best_index = i;
    }
  }
  if (best_index < 0) {
    throw LinoptNativeError("No valid pivot index found.");
  }
  return best_index;
}

Eigen::VectorXi to_eigen_indices(const std::vector<int>& variables) {
  Eigen::VectorXi out(static_cast<int>(variables.size()));
  for (int i = 0; i < out.size(); ++i) {
    out[i] = variables[static_cast<std::size_t>(i)];
  }
  return out;
}

int dual_minimum_ratio_index(
    const std::vector<int>& non_basic_vars,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& pivot_direction) {
  return index_of_smallest_ratio(to_eigen_indices(non_basic_vars), s, pivot_direction);
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

void PrimalPivotingStrategy::initialize(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXi& basis,
    const std::vector<int>& non_basic_vars,
    const Eigen::MatrixXd& inv_basis) {
  (void)a;
  (void)basis;
  (void)non_basic_vars;
  (void)inv_basis;
}

int BlandsRule::pick_entering_index(
    const Eigen::VectorXd& reduced_costs,
    const std::vector<int>& non_basic_vars) {
  int best_index = -1;
  int best_variable = std::numeric_limits<int>::max();
  for (int i = 0; i < reduced_costs.size(); ++i) {
    const int variable = non_basic_vars[static_cast<std::size_t>(i)];
    if (reduced_costs[i] < -kPivotTolerance && variable < best_variable) {
      best_variable = variable;
      best_index = i;
    }
  }
  if (best_index < 0) {
    throw LinoptNativeError("No valid entering variable found.");
  }
  return best_index;
}

int BlandsRule::pick_exiting_index(
    const Eigen::VectorXi& basis,
    const Eigen::VectorXd& x_basis,
    const Eigen::VectorXd& basic_direction,
    const Eigen::MatrixXd& inv_basis) {
  (void)inv_basis;
  return index_of_smallest_ratio(basis, x_basis, basic_direction);
}

int DantzigsRule::pick_entering_index(
    const Eigen::VectorXd& reduced_costs,
    const std::vector<int>& non_basic_vars) {
  (void)non_basic_vars;
  int best_index = -1;
  reduced_costs.minCoeff(&best_index);
  if (best_index < 0) {
    throw LinoptNativeError("No valid entering variable found.");
  }
  return best_index;
}

int DantzigsRule::pick_exiting_index(
    const Eigen::VectorXi& basis,
    const Eigen::VectorXd& x_basis,
    const Eigen::VectorXd& basic_direction,
    const Eigen::MatrixXd& inv_basis) {
  (void)inv_basis;
  return index_of_smallest_ratio(basis, x_basis, basic_direction);
}

void SteepestEdgeRule::initialize(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXi& basis,
    const std::vector<int>& non_basic_vars,
    const Eigen::MatrixXd& inv_basis) {
  (void)basis;
  a_ = &a;
  entering_index_ = -1;
  non_basic_vars_ = non_basic_vars;
  const Eigen::MatrixXd directions = inv_basis * columns(a, non_basic_vars_);
  norm_eta_squared_ = Eigen::VectorXd::Ones(directions.cols()) +
                      directions.array().square().colwise().sum().transpose().matrix();
}

int SteepestEdgeRule::pick_entering_index(
    const Eigen::VectorXd& reduced_costs,
    const std::vector<int>& non_basic_vars) {
  if (non_basic_vars != non_basic_vars_ || norm_eta_squared_.size() != reduced_costs.size()) {
    throw LinoptNativeError("Steepest-edge weights are not aligned with non-basic variables.");
  }

  int best_index = -1;
  double best_score = std::numeric_limits<double>::infinity();
  for (int i = 0; i < reduced_costs.size(); ++i) {
    if (reduced_costs[i] >= -kPivotTolerance) {
      continue;
    }
    const double score =
        reduced_costs[i] / std::sqrt(std::max(norm_eta_squared_[i], kPivotTolerance));
    if (score < best_score) {
      best_score = score;
      best_index = i;
    }
  }
  if (best_index < 0) {
    throw LinoptNativeError("No valid entering variable found.");
  }
  entering_index_ = best_index;
  return best_index;
}

int SteepestEdgeRule::pick_exiting_index(
    const Eigen::VectorXi& basis,
    const Eigen::VectorXd& x_basis,
    const Eigen::VectorXd& basic_direction,
    const Eigen::MatrixXd& inv_basis) {
  if (a_ == nullptr) {
    throw LinoptNativeError("SteepestEdgeRule must be initialized before use.");
  }
  if (entering_index_ < 0 ||
      entering_index_ >= static_cast<int>(non_basic_vars_.size())) {
    throw LinoptNativeError("Entering index is invalid.");
  }

  const int exiting_index = index_of_smallest_ratio(basis, x_basis, basic_direction);
  const int exiting_variable = basis[exiting_index];
  const double pivot = basic_direction[exiting_index];
  if (std::abs(pivot) <= kPivotTolerance) {
    throw LinoptNativeError("Pivot is too close to zero.");
  }

  const Eigen::MatrixXd non_basic_columns = columns(*a_, non_basic_vars_);
  const Eigen::RowVectorXd alpha =
      (inv_basis.row(exiting_index) * non_basic_columns) / pivot;
  const Eigen::RowVectorXd direction_dot_products =
      (basic_direction.transpose() * inv_basis) * non_basic_columns;
  const double entering_gamma = norm_eta_squared_[entering_index_];

  norm_eta_squared_.array() -= 2.0 * alpha.array() * direction_dot_products.array();
  norm_eta_squared_.array() += alpha.array().square() * entering_gamma;
  non_basic_vars_[static_cast<std::size_t>(entering_index_)] = exiting_variable;
  norm_eta_squared_[entering_index_] = entering_gamma / (pivot * pivot);
  norm_eta_squared_ = norm_eta_squared_.array().max(kPivotTolerance).matrix();
  entering_index_ = -1;

  return exiting_index;
}

void DualPivotingStrategy::initialize(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXi& basis,
    const std::vector<int>& non_basic_vars,
    const Eigen::MatrixXd& inv_basis) {
  (void)a;
  (void)basis;
  (void)non_basic_vars;
  (void)inv_basis;
}

int DualBlandsRule::pick_exiting_index(
    const Eigen::VectorXd& x_basis,
    const Eigen::VectorXi& basis,
    const Eigen::MatrixXd& inv_basis) {
  (void)inv_basis;
  int best_index = -1;
  int best_variable = std::numeric_limits<int>::max();
  for (int i = 0; i < x_basis.size(); ++i) {
    if (x_basis[i] < -kPivotTolerance && basis[i] < best_variable) {
      best_variable = basis[i];
      best_index = i;
    }
  }
  if (best_index < 0) {
    throw LinoptNativeError("No valid exiting variable found.");
  }
  return best_index;
}

int DualBlandsRule::pick_entering_index(
    const std::vector<int>& non_basic_vars,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& pivot_direction) {
  return dual_minimum_ratio_index(non_basic_vars, s, pivot_direction);
}

int DualDantzigsRule::pick_exiting_index(
    const Eigen::VectorXd& x_basis,
    const Eigen::VectorXi& basis,
    const Eigen::MatrixXd& inv_basis) {
  (void)basis;
  (void)inv_basis;
  int best_index = -1;
  x_basis.minCoeff(&best_index);
  if (best_index < 0 || x_basis[best_index] >= -kPivotTolerance) {
    throw LinoptNativeError("No valid exiting variable found.");
  }
  return best_index;
}

int DualDantzigsRule::pick_entering_index(
    const std::vector<int>& non_basic_vars,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& pivot_direction) {
  return dual_minimum_ratio_index(non_basic_vars, s, pivot_direction);
}

int DualSteepestEdgeRule::pick_exiting_index(
    const Eigen::VectorXd& x_basis,
    const Eigen::VectorXi& basis,
    const Eigen::MatrixXd& inv_basis) {
  (void)basis;
  int best_index = -1;
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
  if (best_index < 0) {
    throw LinoptNativeError("No valid exiting variable found.");
  }
  return best_index;
}

int DualSteepestEdgeRule::pick_entering_index(
    const std::vector<int>& non_basic_vars,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& pivot_direction) {
  return dual_minimum_ratio_index(non_basic_vars, s, pivot_direction);
}

std::unique_ptr<PrimalPivotingStrategy> make_primal_pivoting_strategy(PivotRule pivot_rule) {
  if (pivot_rule == PivotRule::Bland) {
    return std::make_unique<BlandsRule>();
  }
  if (pivot_rule == PivotRule::Dantzig) {
    return std::make_unique<DantzigsRule>();
  }
  if (pivot_rule == PivotRule::SteepestEdge) {
    return std::make_unique<SteepestEdgeRule>();
  }
  throw LinoptNativeError("Primal simplex requires a primal pivot rule.");
}

std::unique_ptr<DualPivotingStrategy> make_dual_pivoting_strategy(PivotRule pivot_rule) {
  if (pivot_rule == PivotRule::DualBland) {
    return std::make_unique<DualBlandsRule>();
  }
  if (pivot_rule == PivotRule::DualDantzig) {
    return std::make_unique<DualDantzigsRule>();
  }
  if (pivot_rule == PivotRule::DualSteepestEdge) {
    return std::make_unique<DualSteepestEdgeRule>();
  }
  throw LinoptNativeError("Dual simplex requires a dual pivot rule.");
}

Eigen::MatrixXd update_inverse(
    const Eigen::MatrixXd& a,
    const Eigen::MatrixXd& b_inv,
    int entering_variable,
    int exiting_index) {
  const Eigen::VectorXd eta = b_inv * a.col(entering_variable);
  const double pivot = eta[exiting_index];
  if (std::abs(pivot) <= kPivotTolerance) {
    throw LinoptNativeError("Pivot is too close to zero.");
  }
  const Eigen::RowVectorXd pivot_row = b_inv.row(exiting_index) / pivot;
  Eigen::VectorXd eta_minus_unit = eta;
  eta_minus_unit[exiting_index] -= 1.0;
  return b_inv - eta_minus_unit * pivot_row;
}

}  // namespace linopt_native
