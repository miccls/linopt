#include "linopt_native/solvers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace linopt_native {
namespace {

PrimalDualPoint newton_direction(
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

PrimalDualPoint predictor_corrector_direction(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c,
    const PrimalDualPoint& point) {
  const Eigen::VectorXd r_c = a.transpose() * point.lam + point.s - c;
  const Eigen::VectorXd r_b = a * point.x - b;
  const Eigen::VectorXd xs = point.x.array() * point.s.array();

  const PrimalDualPoint newton = newton_direction(a, b, c, point);
  const PrimalDualPoint affine = affine_step(point, newton);
  const double mu = duality_measure(point.x, point.s);
  const double sigma = std::pow(mu_after_step(point, affine) / mu, 3.0);
  const Eigen::VectorXd dxds = newton.x.array() * newton.s.array();
  const Eigen::VectorXd r_xs =
      xs + dxds - sigma * mu * Eigen::VectorXd::Ones(point.x.size());
  return solve_ipm_system_dense(a, point, r_c, r_b, r_xs);
}

std::string ipm_progress_row(
    int iteration,
    double primal_objective,
    double dual_objective,
    double primal_residual,
    double dual_residual,
    double complementarity,
    double elapsed_seconds) {
  std::ostringstream out;
  out << std::setw(4) << iteration << "   " << std::scientific << std::setprecision(3)
      << std::setw(10) << primal_objective << "  " << std::setw(10)
      << dual_objective << "  " << std::setw(10) << primal_residual << " "
      << std::setw(10) << dual_residual << "  " << std::setw(8)
      << complementarity << "    " << std::defaultfloat << std::setprecision(2)
      << std::setw(5) << elapsed_seconds << "s";
  return out.str();
}

double elapsed_seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
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

IpmResult solve_predictor_corrector_dense(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c,
    const PrimalDualPoint& initial_point,
    int max_iterations,
    double optimality_tolerance,
    LogCallback log_callback) {
  IpmResult result;
  result.point = initial_point;

  int iteration = 0;
  const auto start = std::chrono::steady_clock::now();
  while (duality_measure(result.point.x, result.point.s) > optimality_tolerance &&
         iteration < max_iterations) {
    const PrimalDualPoint direction = predictor_corrector_direction(a, b, c, result.point);
    const double eta = std::pow(0.995, 1.0 / static_cast<double>(iteration + 1));
    const double primal_step = std::min(max_step_size(result.point.x, direction.x) * eta, 1.0);
    const double dual_step = std::min(max_step_size(result.point.s, direction.s) * eta, 1.0);

    result.point.x += primal_step * direction.x;
    result.point.lam += dual_step * direction.lam;
    result.point.s += dual_step * direction.s;

    result.primal_objective_history.push_back(c.dot(result.point.x));
    result.dual_objective_history.push_back(b.dot(result.point.lam));
    result.complementarity_history.push_back(duality_measure(result.point.x, result.point.s));
    if (log_callback) {
      log_callback(ipm_progress_row(
          iteration,
          result.primal_objective_history.back(),
          result.dual_objective_history.back(),
          (a * result.point.x - b).norm(),
          (a.transpose() * result.point.lam + result.point.s - c).norm(),
          result.complementarity_history.back(),
          elapsed_seconds(start)));
    }
    ++iteration;
  }

  result.iterations = iteration;
  result.objective = c.dot(result.point.x);
  if (iteration >= max_iterations &&
      duality_measure(result.point.x, result.point.s) > optimality_tolerance) {
    result.status = "iteration_limit";
    result.message = "Predictor-corrector reached the iteration limit.";
  }
  return result;
}

}  // namespace linopt_native
