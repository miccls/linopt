#pragma once

#include "linopt_native/pivot_strategy.hpp"
#include "linopt_native/types.hpp"

#include <memory>

namespace linopt_native {

class PrimalSimplex {
 public:
  explicit PrimalSimplex(PivotRule pivot_rule, LogCallback log_callback = nullptr);

  SimplexResult solve(
      const Eigen::MatrixXd& a,
      const Eigen::VectorXd& b,
      const Eigen::VectorXd& c,
      const Eigen::VectorXi& initial_basis,
      int max_iterations);

 private:
  std::unique_ptr<PrimalPivotingStrategy> pivot_strategy_;
  LogCallback log_callback_;
};

SimplexResult solve_primal_simplex_dense(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c,
    const Eigen::VectorXi& initial_basis,
    int max_iterations,
    PivotRule pivot_rule,
    LogCallback log_callback = nullptr);

}  // namespace linopt_native
