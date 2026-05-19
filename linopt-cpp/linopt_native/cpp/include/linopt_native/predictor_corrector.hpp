#pragma once

#include "linopt_native/types.hpp"

namespace linopt_native {

class PredictorCorrector {
 public:
  PredictorCorrector(
      int max_iterations,
      double optimality_tolerance,
      LogCallback log_callback = nullptr);

  IpmResult solve(
      const Eigen::MatrixXd& a,
      const Eigen::VectorXd& b,
      const Eigen::VectorXd& c,
      const PrimalDualPoint& initial_point) const;

 private:
  int max_iterations_;
  double optimality_tolerance_;
  LogCallback log_callback_;
};

IpmResult solve_predictor_corrector_dense(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c,
    const PrimalDualPoint& initial_point,
    int max_iterations,
    double optimality_tolerance,
    LogCallback log_callback = nullptr);

}  // namespace linopt_native
