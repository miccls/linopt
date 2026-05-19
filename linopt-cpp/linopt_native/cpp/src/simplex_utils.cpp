#include "linopt_native/simplex_utils.hpp"

#include <cstdlib>
#include <iomanip>
#include <set>
#include <sstream>
#include <utility>

namespace linopt_native {

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
  const Eigen::VectorXd full_reduced_costs = c - a.transpose() * lambda;
  return elements(full_reduced_costs, non_basic_vars);
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

bool native_profile_enabled() {
  const char* value = std::getenv("LINOPT_NATIVE_PROFILE");
  return value != nullptr && std::string(value) != "0" && std::string(value) != "";
}

ProfileScope::ProfileScope(NativeProfile* profile, std::string section)
    : profile_(profile != nullptr && profile->enabled ? profile : nullptr),
      section_(std::move(section)),
      start_(std::chrono::steady_clock::now()) {}

ProfileScope::~ProfileScope() {
  if (profile_ == nullptr) {
    return;
  }
  const double seconds = elapsed_seconds(start_);
  profile_->seconds[section_] += seconds;
  profile_->calls[section_] += 1;
}

}  // namespace linopt_native
