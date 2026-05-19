#pragma once

#include "linopt_native/types.hpp"

namespace linopt_native {

double duality_measure(const Eigen::VectorXd& x, const Eigen::VectorXd& s);

double max_step_size(const Eigen::VectorXd& x, const Eigen::VectorXd& dx);

PrimalDualPoint solve_ipm_system_dense(
    const Eigen::MatrixXd& a,
    const PrimalDualPoint& point,
    const Eigen::VectorXd& r_c,
    const Eigen::VectorXd& r_b,
    const Eigen::VectorXd& r_xs);

PrimalDualPoint solve_predictor_corrector_direction(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c,
    const PrimalDualPoint& point);

}  // namespace linopt_native
