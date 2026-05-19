#include "linopt_native/solvers.hpp"

#include <gtest/gtest.h>

TEST(NativeIpm, StepAndDualityPrimitives) {
  Eigen::VectorXd x(3);
  x << 1.0, 2.0, 3.0;
  Eigen::VectorXd s(3);
  s << 0.0, 2.0, 0.0;
  EXPECT_NEAR(linopt_native::duality_measure(x, s), 4.0 / 3.0, 1e-12);

  Eigen::VectorXd dx(4);
  Eigen::VectorXd y(4);
  y << 4.0, 6.0, 8.0, 9.0;
  dx << 0.0, -1.0, -1000.0, 10.0;
  EXPECT_NEAR(linopt_native::max_step_size(y, dx), 0.008, 1e-12);
}

TEST(NativeIpm, SolvesUnitBoxLp) {
  Eigen::MatrixXd a(2, 4);
  a << 1.0, 0.0, 1.0, 0.0,
       0.0, 1.0, 0.0, 1.0;
  Eigen::VectorXd b(2);
  b << 1.0, 1.0;
  Eigen::VectorXd c(4);
  c << 1.0, 1.0, 0.0, 0.0;
  linopt_native::PrimalDualPoint point{
      Eigen::VectorXd::Constant(4, 0.5),
      Eigen::VectorXd::Constant(2, -0.25),
      (Eigen::VectorXd(4) << 1.25, 1.25, 0.25, 0.25).finished(),
  };

  const auto result = linopt_native::solve_predictor_corrector_dense(
      a, b, c, point, 100, 1e-10);

  EXPECT_EQ(result.status, "optimal");
  Eigen::VectorXd expected(4);
  expected << 0.0, 0.0, 1.0, 1.0;
  EXPECT_TRUE(result.point.x.isApprox(expected, 1e-6));
}
