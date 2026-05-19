#pragma once

#include "linopt_native/types.hpp"

#include <chrono>
#include <string>
#include <vector>

namespace linopt_native {

Eigen::MatrixXd columns(const Eigen::MatrixXd& a, const Eigen::VectorXi& indices);
Eigen::MatrixXd columns(const Eigen::MatrixXd& a, const std::vector<int>& indices);
Eigen::VectorXd elements(const Eigen::VectorXd& values, const Eigen::VectorXi& indices);
Eigen::VectorXd elements(const Eigen::VectorXd& values, const std::vector<int>& indices);
Eigen::VectorXd reduced_costs(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& c,
    const Eigen::VectorXi& basis,
    const std::vector<int>& non_basic_vars,
    const Eigen::MatrixXd& inv_basis);
void append_history(SimplexResult& result, const Eigen::VectorXi& basis, double objective);
void validate_basis(const Eigen::MatrixXd& a, const Eigen::VectorXi& basis);
std::string simplex_progress_row(
    int iteration,
    double objective,
    double primal_inf,
    double dual_inf,
    double elapsed_seconds);
std::string initial_objective_message(double objective);
std::string optimal_message(double objective, int iterations);
std::string iteration_limit_message(int max_iterations);
double elapsed_seconds(std::chrono::steady_clock::time_point start);
bool native_profile_enabled();

class ProfileScope {
 public:
  ProfileScope(NativeProfile* profile, std::string section);
  ~ProfileScope();

 private:
  NativeProfile* profile_;
  std::string section_;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace linopt_native
