#include "linopt_native/solvers.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

linopt_native::LogCallback make_log_callback(const py::object& callback) {
  if (callback.is_none()) {
    return nullptr;
  }
  return [callback](const std::string& message) {
    py::gil_scoped_acquire acquire;
    callback(message);
  };
}

py::list basis_history_to_py(const std::vector<Eigen::VectorXi>& history) {
  py::list out;
  for (const Eigen::VectorXi& basis : history) {
    out.append(basis);
  }
  return out;
}

py::dict simplex_result_to_py(const linopt_native::SimplexResult& result) {
  py::dict out;
  out["status"] = result.status;
  out["message"] = result.message;
  out["basis"] = result.basis;
  out["solution"] = result.solution;
  out["objective"] = result.objective;
  out["iterations"] = result.iterations;
  out["basis_history"] = basis_history_to_py(result.basis_history);
  out["objective_history"] = result.objective_history;
  out["profile_enabled"] = result.profile.enabled;
  out["profile_seconds"] = result.profile.seconds;
  out["profile_calls"] = result.profile.calls;
  return out;
}

py::dict ipm_result_to_py(const linopt_native::IpmResult& result) {
  py::dict out;
  out["status"] = result.status;
  out["message"] = result.message;
  out["x"] = result.point.x;
  out["lam"] = result.point.lam;
  out["s"] = result.point.s;
  out["objective"] = result.objective;
  out["iterations"] = result.iterations;
  out["primal_objective_history"] = result.primal_objective_history;
  out["dual_objective_history"] = result.dual_objective_history;
  out["complementarity_history"] = result.complementarity_history;
  return out;
}

}  // namespace

PYBIND11_MODULE(_native, m) {
  py::register_exception<linopt_native::LinoptNativeError>(m, "LinoptNativeError");

  py::enum_<linopt_native::PivotRule>(m, "PivotRule")
      .value("Bland", linopt_native::PivotRule::Bland)
      .value("Dantzig", linopt_native::PivotRule::Dantzig)
      .value("SteepestEdge", linopt_native::PivotRule::SteepestEdge)
      .value("DualBland", linopt_native::PivotRule::DualBland)
      .value("DualDantzig", linopt_native::PivotRule::DualDantzig)
      .value("DualSteepestEdge", linopt_native::PivotRule::DualSteepestEdge);

  m.def(
      "solve_primal_simplex_dense",
      [](const Eigen::MatrixXd& a,
         const Eigen::VectorXd& b,
         const Eigen::VectorXd& c,
         const Eigen::VectorXi& initial_basis,
         int max_iterations,
         linopt_native::PivotRule pivot_rule,
         const py::object& log_callback) {
        return simplex_result_to_py(linopt_native::solve_primal_simplex_dense(
            a,
            b,
            c,
            initial_basis,
            max_iterations,
            pivot_rule,
            make_log_callback(log_callback)));
      },
      py::arg("A"),
      py::arg("b"),
      py::arg("c"),
      py::arg("initial_basis"),
      py::arg("max_iterations"),
      py::arg("pivot_rule"),
      py::arg("log_callback") = py::none());

  m.def(
      "solve_dual_simplex_dense",
      [](const Eigen::MatrixXd& a,
         const Eigen::VectorXd& b,
         const Eigen::VectorXd& c,
         const Eigen::VectorXi& initial_basis,
         int max_iterations,
         linopt_native::PivotRule pivot_rule,
         const py::object& log_callback) {
        return simplex_result_to_py(linopt_native::solve_dual_simplex_dense(
            a,
            b,
            c,
            initial_basis,
            max_iterations,
            pivot_rule,
            make_log_callback(log_callback)));
      },
      py::arg("A"),
      py::arg("b"),
      py::arg("c"),
      py::arg("initial_basis"),
      py::arg("max_iterations"),
      py::arg("pivot_rule"),
      py::arg("log_callback") = py::none());

  m.def(
      "solve_predictor_corrector_dense",
      [](const Eigen::MatrixXd& a,
         const Eigen::VectorXd& b,
         const Eigen::VectorXd& c,
         const py::dict& initial_point,
         int max_iterations,
         double optimality_tolerance,
         const py::object& log_callback) {
        linopt_native::PrimalDualPoint point{
            initial_point["x"].cast<Eigen::VectorXd>(),
            initial_point["lam"].cast<Eigen::VectorXd>(),
            initial_point["s"].cast<Eigen::VectorXd>(),
        };
        return ipm_result_to_py(linopt_native::solve_predictor_corrector_dense(
            a,
            b,
            c,
            point,
            max_iterations,
            optimality_tolerance,
            make_log_callback(log_callback)));
      },
      py::arg("A"),
      py::arg("b"),
      py::arg("c"),
      py::arg("initial_point"),
      py::arg("max_iterations"),
      py::arg("optimality_tolerance"),
      py::arg("log_callback") = py::none());
}
