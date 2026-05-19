#include "linopt_native/predictor_corrector.hpp"

#include "linopt_native/ipm_tools.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>

namespace linopt_native {
namespace {

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

PredictorCorrector::PredictorCorrector(
    int max_iterations,
    double optimality_tolerance,
    LogCallback log_callback)
    : max_iterations_(max_iterations),
      optimality_tolerance_(optimality_tolerance),
      log_callback_(std::move(log_callback)) {}

IpmResult PredictorCorrector::solve(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c,
    const PrimalDualPoint& initial_point) const {
  IpmResult result;
  result.point = initial_point;

  int iteration = 0;
  const auto start = std::chrono::steady_clock::now();
  while (duality_measure(result.point.x, result.point.s) > optimality_tolerance_ &&
         iteration < max_iterations_) {
    const PrimalDualPoint direction = solve_predictor_corrector_direction(a, b, c, result.point);
    const double eta = std::pow(0.995, 1.0 / static_cast<double>(iteration + 1));
    const double primal_step = std::min(max_step_size(result.point.x, direction.x) * eta, 1.0);
    const double dual_step = std::min(max_step_size(result.point.s, direction.s) * eta, 1.0);

    result.point.x += primal_step * direction.x;
    result.point.lam += dual_step * direction.lam;
    result.point.s += dual_step * direction.s;

    result.primal_objective_history.push_back(c.dot(result.point.x));
    result.dual_objective_history.push_back(b.dot(result.point.lam));
    result.complementarity_history.push_back(duality_measure(result.point.x, result.point.s));
    if (log_callback_) {
      log_callback_(ipm_progress_row(
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
  if (iteration >= max_iterations_ &&
      duality_measure(result.point.x, result.point.s) > optimality_tolerance_) {
    result.status = "iteration_limit";
    result.message = "Predictor-corrector reached the iteration limit.";
  }
  return result;
}

IpmResult solve_predictor_corrector_dense(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c,
    const PrimalDualPoint& initial_point,
    int max_iterations,
    double optimality_tolerance,
    LogCallback log_callback) {
  PredictorCorrector solver(max_iterations, optimality_tolerance, std::move(log_callback));
  return solver.solve(a, b, c, initial_point);
}

}  // namespace linopt_native
