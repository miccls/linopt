#include "linopt_native/solvers.hpp"

#include <gtest/gtest.h>

TEST(NativeSimplex, PrimalPivotRulesSelectExpectedPositions) {
  Eigen::VectorXd reduced_costs(7);
  reduced_costs << 0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0;
  std::vector<int> non_basic_vars{0, 4, 5, 6, 3, 1, 2};

  linopt_native::BlandsRule bland;
  linopt_native::DantzigsRule dantzig;

  EXPECT_EQ(bland.pick_entering_index(reduced_costs, non_basic_vars), 5);
  EXPECT_EQ(dantzig.pick_entering_index(reduced_costs, non_basic_vars), 6);
}

TEST(NativeSimplex, InverseUpdateMatchesRecomputedInverse) {
  Eigen::MatrixXd a(2, 4);
  a << 1.0, 2.0, 3.0, 4.0,
       4.0, 3.0, 2.0, 1.0;
  Eigen::VectorXi basis(2);
  basis << 1, 2;
  Eigen::MatrixXd b_inv = a(Eigen::all, basis).inverse();

  basis[1] = 3;
  Eigen::MatrixXd updated = linopt_native::update_inverse(a, b_inv, 3, 1);
  Eigen::MatrixXd expected = a(Eigen::all, basis).inverse();

  EXPECT_TRUE(updated.isApprox(expected));
}

TEST(NativeSimplex, SolvesSmallPrimalLp) {
  Eigen::MatrixXd a(3, 6);
  a << 1, 2, 2, 1, 0, 0,
       2, 1, 2, 0, 1, 0,
       2, 2, 1, 0, 0, 1;
  Eigen::VectorXd b(3);
  b << 20, 20, 20;
  Eigen::VectorXd c(6);
  c << -10, -12, -12, 0, 0, 0;
  Eigen::VectorXi basis(3);
  basis << 3, 4, 5;

  const auto result = linopt_native::solve_primal_simplex_dense(
      a, b, c, basis, 100, linopt_native::PivotRule::Bland);

  EXPECT_EQ(result.status, "optimal");
  EXPECT_NEAR(result.objective, -136.0, 1e-8);
  Eigen::VectorXd expected(6);
  expected << 4, 4, 4, 0, 0, 0;
  EXPECT_TRUE(result.solution.isApprox(expected));
}

TEST(NativeSimplex, PrimalSolverClassSolvesSmallLp) {
  Eigen::MatrixXd a(3, 6);
  a << 1, 2, 2, 1, 0, 0,
       2, 1, 2, 0, 1, 0,
       2, 2, 1, 0, 0, 1;
  Eigen::VectorXd b(3);
  b << 20, 20, 20;
  Eigen::VectorXd c(6);
  c << -10, -12, -12, 0, 0, 0;
  Eigen::VectorXi basis(3);
  basis << 3, 4, 5;

  linopt_native::PrimalSimplex solver(linopt_native::PivotRule::Dantzig);
  const auto result = solver.solve(a, b, c, basis, 100);

  EXPECT_EQ(result.status, "optimal");
  EXPECT_NEAR(result.objective, -136.0, 1e-8);
}

TEST(NativeSimplex, SolvesSmallDualLp) {
  Eigen::MatrixXd a(4, 7);
  a << 1, 2, 2, 1, 0, 0, 0,
       2, 1, 2, 0, 1, 0, 0,
       2, 2, 1, 0, 0, 1, 0,
       1, 3, 3, 0, 0, 0, 1;
  Eigen::VectorXd b(4);
  b << 20, 20, 20, 4;
  Eigen::VectorXd c(7);
  c << -10, -12, -12, 0, 0, 0, 0;
  Eigen::VectorXi basis(4);
  basis << 0, 1, 2, 6;

  const auto result = linopt_native::solve_dual_simplex_dense(
      a, b, c, basis, 100, linopt_native::PivotRule::DualBland);

  EXPECT_EQ(result.status, "optimal");
  EXPECT_NEAR(result.objective, -40.0, 1e-8);
}

TEST(NativeSimplex, DualSolverClassSolvesSmallLp) {
  Eigen::MatrixXd a(4, 7);
  a << 1, 2, 2, 1, 0, 0, 0,
       2, 1, 2, 0, 1, 0, 0,
       2, 2, 1, 0, 0, 1, 0,
       1, 3, 3, 0, 0, 0, 1;
  Eigen::VectorXd b(4);
  b << 20, 20, 20, 4;
  Eigen::VectorXd c(7);
  c << -10, -12, -12, 0, 0, 0, 0;
  Eigen::VectorXi basis(4);
  basis << 0, 1, 2, 6;

  linopt_native::DualSimplex solver(linopt_native::PivotRule::DualDantzig);
  const auto result = solver.solve(a, b, c, basis, 100);

  EXPECT_EQ(result.status, "optimal");
  EXPECT_NEAR(result.objective, -40.0, 1e-8);
}
