#pragma once

#include "linopt_native/types.hpp"

#include <memory>
#include <vector>

namespace linopt_native {

std::vector<int> non_basic_variables(int num_variables, const Eigen::VectorXi& basis);

class PrimalPivotingStrategy {
 public:
  virtual ~PrimalPivotingStrategy() = default;

  virtual void initialize(
      const Eigen::MatrixXd& a,
      const Eigen::VectorXi& basis,
      const std::vector<int>& non_basic_vars,
      const Eigen::MatrixXd& inv_basis);

  virtual int pick_entering_index(
      const Eigen::VectorXd& reduced_costs,
      const std::vector<int>& non_basic_vars) = 0;

  virtual int pick_exiting_index(
      const Eigen::VectorXi& basis,
      const Eigen::VectorXd& x_basis,
      const Eigen::VectorXd& basic_direction,
      const Eigen::MatrixXd& inv_basis) = 0;
};

class BlandsRule final : public PrimalPivotingStrategy {
 public:
  int pick_entering_index(
      const Eigen::VectorXd& reduced_costs,
      const std::vector<int>& non_basic_vars) override;

  int pick_exiting_index(
      const Eigen::VectorXi& basis,
      const Eigen::VectorXd& x_basis,
      const Eigen::VectorXd& basic_direction,
      const Eigen::MatrixXd& inv_basis) override;
};

class DantzigsRule final : public PrimalPivotingStrategy {
 public:
  int pick_entering_index(
      const Eigen::VectorXd& reduced_costs,
      const std::vector<int>& non_basic_vars) override;

  int pick_exiting_index(
      const Eigen::VectorXi& basis,
      const Eigen::VectorXd& x_basis,
      const Eigen::VectorXd& basic_direction,
      const Eigen::MatrixXd& inv_basis) override;
};

class SteepestEdgeRule final : public PrimalPivotingStrategy {
 public:
  void initialize(
      const Eigen::MatrixXd& a,
      const Eigen::VectorXi& basis,
      const std::vector<int>& non_basic_vars,
      const Eigen::MatrixXd& inv_basis) override;

  int pick_entering_index(
      const Eigen::VectorXd& reduced_costs,
      const std::vector<int>& non_basic_vars) override;

  int pick_exiting_index(
      const Eigen::VectorXi& basis,
      const Eigen::VectorXd& x_basis,
      const Eigen::VectorXd& basic_direction,
      const Eigen::MatrixXd& inv_basis) override;

 private:
  const Eigen::MatrixXd* a_ = nullptr;
  int entering_index_ = -1;
  std::vector<int> non_basic_vars_;
  Eigen::VectorXd norm_eta_squared_;
};

class DualPivotingStrategy {
 public:
  virtual ~DualPivotingStrategy() = default;

  virtual void initialize(
      const Eigen::MatrixXd& a,
      const Eigen::VectorXi& basis,
      const std::vector<int>& non_basic_vars,
      const Eigen::MatrixXd& inv_basis);

  virtual int pick_exiting_index(
      const Eigen::VectorXd& x_basis,
      const Eigen::VectorXi& basis,
      const Eigen::MatrixXd& inv_basis) = 0;

  virtual int pick_entering_index(
      const std::vector<int>& non_basic_vars,
      const Eigen::VectorXd& s,
      const Eigen::VectorXd& pivot_direction) = 0;
};

class DualBlandsRule final : public DualPivotingStrategy {
 public:
  int pick_exiting_index(
      const Eigen::VectorXd& x_basis,
      const Eigen::VectorXi& basis,
      const Eigen::MatrixXd& inv_basis) override;

  int pick_entering_index(
      const std::vector<int>& non_basic_vars,
      const Eigen::VectorXd& s,
      const Eigen::VectorXd& pivot_direction) override;
};

class DualDantzigsRule final : public DualPivotingStrategy {
 public:
  int pick_exiting_index(
      const Eigen::VectorXd& x_basis,
      const Eigen::VectorXi& basis,
      const Eigen::MatrixXd& inv_basis) override;

  int pick_entering_index(
      const std::vector<int>& non_basic_vars,
      const Eigen::VectorXd& s,
      const Eigen::VectorXd& pivot_direction) override;
};

class DualSteepestEdgeRule final : public DualPivotingStrategy {
 public:
  int pick_exiting_index(
      const Eigen::VectorXd& x_basis,
      const Eigen::VectorXi& basis,
      const Eigen::MatrixXd& inv_basis) override;

  int pick_entering_index(
      const std::vector<int>& non_basic_vars,
      const Eigen::VectorXd& s,
      const Eigen::VectorXd& pivot_direction) override;
};

std::unique_ptr<PrimalPivotingStrategy> make_primal_pivoting_strategy(PivotRule pivot_rule);
std::unique_ptr<DualPivotingStrategy> make_dual_pivoting_strategy(PivotRule pivot_rule);

Eigen::MatrixXd update_inverse(
    const Eigen::MatrixXd& a,
    const Eigen::MatrixXd& b_inv,
    int entering_variable,
    int exiting_index);

}  // namespace linopt_native
