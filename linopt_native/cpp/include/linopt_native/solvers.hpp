#pragma once

#include <Eigen/Dense>

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace linopt_native {

constexpr double kPivotTolerance = 1e-5;
constexpr double kNonNegativityTolerance = 1e-5;
constexpr int kInverseRecomputeInterval = 50;

using LogCallback = std::function<void(const std::string&)>;

enum class PivotRule {
  Bland,
  Dantzig,
  SteepestEdge,
  DualBland,
  DualDantzig,
  DualSteepestEdge,
};

struct SimplexResult {
  std::string status = "optimal";
  std::string message;
  Eigen::VectorXi basis;
  Eigen::VectorXd solution;
  double objective = 0.0;
  int iterations = 0;
  std::vector<Eigen::VectorXi> basis_history;
  std::vector<double> objective_history;
};

struct PrimalDualPoint {
  Eigen::VectorXd x;
  Eigen::VectorXd lam;
  Eigen::VectorXd s;
};

struct IpmResult {
  std::string status = "optimal";
  std::string message;
  PrimalDualPoint point;
  double objective = 0.0;
  int iterations = 0;
  std::vector<double> primal_objective_history;
  std::vector<double> dual_objective_history;
  std::vector<double> complementarity_history;
};

class LinoptNativeError : public std::runtime_error {
 public:
  explicit LinoptNativeError(const std::string& message) : std::runtime_error(message) {}
};

SimplexResult solve_primal_simplex_dense(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c,
    const Eigen::VectorXi& initial_basis,
    int max_iterations,
    PivotRule pivot_rule,
    LogCallback log_callback = nullptr);

SimplexResult solve_dual_simplex_dense(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c,
    const Eigen::VectorXi& initial_basis,
    int max_iterations,
    PivotRule pivot_rule,
    LogCallback log_callback = nullptr);

IpmResult solve_predictor_corrector_dense(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c,
    const PrimalDualPoint& initial_point,
    int max_iterations,
    double optimality_tolerance,
    LogCallback log_callback = nullptr);

Eigen::MatrixXd update_inverse(
    const Eigen::MatrixXd& a,
    const Eigen::MatrixXd& b_inv,
    int entering_variable,
    int exiting_index);

std::vector<int> non_basic_variables(int num_variables, const Eigen::VectorXi& basis);
int primal_entering_index(
    const Eigen::VectorXd& reduced_costs,
    const std::vector<int>& non_basic_vars,
    PivotRule pivot_rule,
    const Eigen::VectorXd* steepest_weights = nullptr);
int primal_exiting_index(
    const Eigen::VectorXi& basis,
    const Eigen::VectorXd& x_basis,
    const Eigen::VectorXd& basic_direction);
int dual_exiting_index(
    const Eigen::VectorXd& x_basis,
    const Eigen::VectorXi& basis,
    const Eigen::MatrixXd& inv_basis,
    PivotRule pivot_rule);
int dual_entering_index(
    const std::vector<int>& non_basic_vars,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& pivot_direction);

double duality_measure(const Eigen::VectorXd& x, const Eigen::VectorXd& s);
double max_step_size(const Eigen::VectorXd& x, const Eigen::VectorXd& dx);
PrimalDualPoint solve_ipm_system_dense(
    const Eigen::MatrixXd& a,
    const PrimalDualPoint& point,
    const Eigen::VectorXd& r_c,
    const Eigen::VectorXd& r_b,
    const Eigen::VectorXd& r_xs);

}  // namespace linopt_native
