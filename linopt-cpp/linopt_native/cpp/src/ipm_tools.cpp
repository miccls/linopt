#include "linopt_native/ipm_tools.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace linopt_native {
namespace {

PrimalDualPoint solve_newton_direction(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c,
    const PrimalDualPoint& point) {
  const Eigen::VectorXd r_c = a.transpose() * point.lam + point.s - c;
  const Eigen::VectorXd r_b = a * point.x - b;
  const Eigen::VectorXd r_xs = point.x.array() * point.s.array();
  return solve_ipm_system_dense(a, point, r_c, r_b, r_xs);
}

PrimalDualPoint affine_step(const PrimalDualPoint& point, const PrimalDualPoint& direction) {
  const double alpha_primal = std::min(1.0, max_step_size(point.x, direction.x));
  const double alpha_dual = std::min(1.0, max_step_size(point.s, direction.s));
  return PrimalDualPoint{
      direction.x * alpha_primal,
      direction.lam * alpha_dual,
      direction.s * alpha_dual,
  };
}

double mu_after_step(const PrimalDualPoint& point, const PrimalDualPoint& step) {
  return duality_measure(point.x + step.x, point.s + step.s);
}

}  // namespace

double duality_measure(const Eigen::VectorXd& x, const Eigen::VectorXd& s) {
  return (x.array() * s.array()).mean();
}

double max_step_size(const Eigen::VectorXd& x, const Eigen::VectorXd& dx) {
  double best = std::numeric_limits<double>::infinity();
  bool has_bound = false;
  for (int i = 0; i < x.size(); ++i) {
    if (dx[i] < 0.0) {
      best = std::min(best, -x[i] / dx[i]);
      has_bound = true;
    }
  }
  return has_bound ? best : 0.0;
}

PrimalDualPoint solve_ipm_system_dense(
    const Eigen::MatrixXd& a,
    const PrimalDualPoint& point,
    const Eigen::VectorXd& r_c,
    const Eigen::VectorXd& r_b,
    const Eigen::VectorXd& r_xs) {
  const Eigen::VectorXd d2 = point.x.array() / point.s.array();
  const Eigen::MatrixXd ad2 = a * d2.asDiagonal();
  const Eigen::MatrixXd normal = ad2 * a.transpose();
  const Eigen::VectorXd rhs =
      -r_b - ad2 * r_c + a * (r_xs.array() / point.s.array()).matrix();
  const Eigen::VectorXd dlam = normal.ldlt().solve(rhs);
  const Eigen::VectorXd ds = -r_c - a.transpose() * dlam;
  const Eigen::VectorXd dx =
      (-(r_xs.array() / point.s.array()) - d2.array() * ds.array()).matrix();
  return PrimalDualPoint{dx, dlam, ds};
}

PrimalDualPoint solve_predictor_corrector_direction(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c,
    const PrimalDualPoint& point) {
  const Eigen::VectorXd r_c = a.transpose() * point.lam + point.s - c;
  const Eigen::VectorXd r_b = a * point.x - b;
  const Eigen::VectorXd xs = point.x.array() * point.s.array();

  const PrimalDualPoint newton = solve_newton_direction(a, b, c, point);
  const PrimalDualPoint affine = affine_step(point, newton);
  const double mu = duality_measure(point.x, point.s);
  const double sigma = std::pow(mu_after_step(point, affine) / mu, 3.0);
  const Eigen::VectorXd dxds = newton.x.array() * newton.s.array();
  const Eigen::VectorXd r_xs =
      xs + dxds - sigma * mu * Eigen::VectorXd::Ones(point.x.size());
  return solve_ipm_system_dense(a, point, r_c, r_b, r_xs);
}

}  // namespace linopt_native
