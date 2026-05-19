#pragma once

#include <Eigen/Dense>

#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace linopt_native {

constexpr double kPivotTolerance = 1e-5;
constexpr double kNonNegativityTolerance = 1e-5;
constexpr int kInverseRecomputeInterval = 50;

using LogCallback = std::function<void(const std::string&)>;

enum class PivotRule {
  Bland,
  Dantzig,
  SteepestEdge,
  DualBland,
  DualDantzig,
  DualSteepestEdge,
};

struct NativeProfile {
  bool enabled = false;
  std::map<std::string, double> seconds;
  std::map<std::string, int> calls;
};

struct SimplexResult {
  std::string status = "optimal";
  std::string message;
  Eigen::VectorXi basis;
  Eigen::VectorXd solution;
  double objective = 0.0;
  int iterations = 0;
  std::vector<Eigen::VectorXi> basis_history;
  std::vector<double> objective_history;
  NativeProfile profile;
};

struct PrimalDualPoint {
  Eigen::VectorXd x;
  Eigen::VectorXd lam;
  Eigen::VectorXd s;
};

struct IpmResult {
  std::string status = "optimal";
  std::string message;
  PrimalDualPoint point;
  double objective = 0.0;
  int iterations = 0;
  std::vector<double> primal_objective_history;
  std::vector<double> dual_objective_history;
  std::vector<double> complementarity_history;
  NativeProfile profile;
};

class LinoptNativeError : public std::runtime_error {
 public:
  explicit LinoptNativeError(const std::string& message) : std::runtime_error(message) {}
};

}  // namespace linopt_native
