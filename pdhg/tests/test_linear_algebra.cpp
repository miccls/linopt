#include <gtest/gtest.h>

#include <cmath>
#include <concepts>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "linear_algebra.h"

namespace pdhg::linalg {
namespace {

void ExpectVectorNear(const Vector& actual, const std::vector<Scalar>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-12);
  }
}

template <typename Exception, std::invocable Callable>
void ExpectThrowsWithMessage(Callable&& callable, std::string_view expected_message) {
  try {
    std::forward<Callable>(callable)();
    FAIL() << "Expected exception";
  } catch (const Exception& exception) {
    EXPECT_EQ(exception.what(), expected_message);
  } catch (...) {
    FAIL() << "Expected a different exception type";
  }
}

TEST(Vector, SupportsBasicPdhgOperations) {
  Vector y{1.0, -2.0, 3.0};
  const Vector x{4.0, 5.0, -6.0};

  EXPECT_DOUBLE_EQ(y.dot(x), -24.0);
  EXPECT_DOUBLE_EQ(y.norm2(), std::sqrt(14.0));
  EXPECT_DOUBLE_EQ(y.minValue(), -2.0);
  EXPECT_DOUBLE_EQ(y.maxValue(), 3.0);

  y.axpy(0.5, x);
  ExpectVectorNear(y, {3.0, 0.5, 0.0});

  y.scale(2.0);
  ExpectVectorNear(y, {6.0, 1.0, 0.0});

  y.clamp(-1.0, 2.0);
  ExpectVectorNear(y, {2.0, 1.0, 0.0});

  y.fill(-3.0);
  ExpectVectorNear(y, {-3.0, -3.0, -3.0});
}

TEST(Vector, RejectsInvalidOperations) {
  Vector x{1.0, 2.0};
  const Vector short_vector{1.0};

  ExpectThrowsWithMessage<std::invalid_argument>([&] { x.axpy(1.0, short_vector); },
                                                 "axpy requires equal vector sizes");
  ExpectThrowsWithMessage<std::invalid_argument>([&] { (void)x.dot(short_vector); },
                                                 "dot requires equal vector sizes");
  ExpectThrowsWithMessage<std::invalid_argument>(
      [&] { x.clamp(2.0, 1.0); }, "clamp lower bound exceeds upper bound");
  ExpectThrowsWithMessage<std::invalid_argument>([] { (void)Vector{}.minValue(); },
                                                 "minValue requires a nonempty vector");
  ExpectThrowsWithMessage<std::invalid_argument>([] { (void)Vector{}.maxValue(); },
                                                 "maxValue requires a nonempty vector");
}

TEST(CooMatrix, ValidatesDimensionsAndIndices) {
  ExpectThrowsWithMessage<std::invalid_argument>([] { (void)CooMatrix(-1, 3); },
                                                 "rows must be nonnegative");
  ExpectThrowsWithMessage<std::invalid_argument>([] { (void)CooMatrix(2, -3); },
                                                 "cols must be nonnegative");

  CooMatrix matrix(2, 3);
  ExpectThrowsWithMessage<std::out_of_range>([&] { matrix.addEntry(-1, 0, 1.0); },
                                             "row is outside matrix bounds");
  ExpectThrowsWithMessage<std::out_of_range>([&] { matrix.addEntry(0, -1, 1.0); },
                                             "col is outside matrix bounds");
  ExpectThrowsWithMessage<std::out_of_range>([&] { matrix.addEntry(2, 0, 1.0); },
                                             "row is outside matrix bounds");
  ExpectThrowsWithMessage<std::out_of_range>([&] { matrix.addEntry(0, 3, 1.0); },
                                             "col is outside matrix bounds");
}

TEST(SparseMatrix, ConvertsCooToCsrWithSortedCombinedEntries) {
  // Matrix:
  //   [0 5 0 2]
  //   [3 0 0 0]
  //   [0 0 4 0]
  CooMatrix coo(3, 4);
  coo.addEntry(0, 3, 2.0);
  coo.addEntry(0, 1, 1.5);
  coo.addEntry(2, 2, 4.0);
  coo.addEntry(1, 0, 3.0);
  coo.addEntry(0, 1, 3.5);
  coo.addEntry(1, 2, 7.0);
  coo.addEntry(1, 2, -7.0);

  const CsrMatrix csr = coo.toCsr();

  EXPECT_EQ(csr.rows(), 3);
  EXPECT_EQ(csr.cols(), 4);
  EXPECT_EQ(csr.nnz(), 4);
  EXPECT_EQ(csr.rowOffset(), (std::vector<Index>{0, 2, 3, 4}));
  EXPECT_EQ(csr.colIdx(), (std::vector<Index>{1, 3, 0, 2}));
  EXPECT_EQ(csr.values(), (std::vector<Scalar>{5.0, 2.0, 3.0, 4.0}));
}

TEST(SparseMatrix, ConvertsCooToCscWithSortedCombinedEntries) {
  // Same matrix as the CSR test, now compressed by column.
  CooMatrix coo(3, 4);
  coo.addEntry(0, 3, 2.0);
  coo.addEntry(0, 1, 1.5);
  coo.addEntry(2, 2, 4.0);
  coo.addEntry(1, 0, 3.0);
  coo.addEntry(0, 1, 3.5);
  coo.addEntry(1, 2, 7.0);
  coo.addEntry(1, 2, -7.0);

  const CscMatrix csc = coo.toCsc();

  EXPECT_EQ(csc.rows(), 3);
  EXPECT_EQ(csc.cols(), 4);
  EXPECT_EQ(csc.nnz(), 4);
  EXPECT_EQ(csc.colOffset(), (std::vector<Index>{0, 1, 2, 3, 4}));
  EXPECT_EQ(csc.rowIdx(), (std::vector<Index>{1, 0, 2, 0}));
  EXPECT_EQ(csc.values(), (std::vector<Scalar>{3.0, 5.0, 4.0, 2.0}));
}

TEST(SparseMatrix, HandlesEmptyRowsAndColumns) {
  CooMatrix coo(4, 5);
  coo.addEntry(0, 2, 1.0);
  coo.addEntry(3, 4, 2.0);

  const CsrMatrix csr = coo.toCsr();
  const CscMatrix csc = coo.toCsc();

  EXPECT_EQ(csr.rowOffset(), (std::vector<Index>{0, 1, 1, 1, 2}));
  EXPECT_EQ(csr.colIdx(), (std::vector<Index>{2, 4}));
  EXPECT_EQ(csc.colOffset(), (std::vector<Index>{0, 0, 0, 1, 1, 2}));
  EXPECT_EQ(csc.rowIdx(), (std::vector<Index>{0, 3}));
}

TEST(SparseMatrix, ComputesCsrMatvec) {
  CooMatrix coo(3, 4);

  // [0 5 0 2]
  // [3 0 0 0]
  // [0 0 4 0]
  coo.addEntry(0, 1, 5.0);
  coo.addEntry(0, 3, 2.0);
  coo.addEntry(1, 0, 3.0);
  coo.addEntry(2, 2, 4.0);

  const CsrMatrix csr = coo.toCsr();
  const Vector x{10.0, 20.0, 30.0, 40.0};

  ExpectVectorNear(csrMatvec(csr, x), {180.0, 30.0, 120.0});

  Vector y(3);
  csrMatvec(csr, x, y);
  ExpectVectorNear(y, {180.0, 30.0, 120.0});
}

TEST(SparseMatrix, ComputesCscTransposeMatvec) {
  CooMatrix coo(3, 4);
  coo.addEntry(0, 1, 5.0);
  coo.addEntry(0, 3, 2.0);
  coo.addEntry(1, 0, 3.0);
  coo.addEntry(2, 2, 4.0);

  const CscMatrix csc = coo.toCsc();
  const Vector y{7.0, 11.0, 13.0};

  ExpectVectorNear(cscTransposeMatvec(csc, y), {33.0, 35.0, 52.0, 14.0});

  Vector x(4);
  cscTransposeMatvec(csc, y, x);
  ExpectVectorNear(x, {33.0, 35.0, 52.0, 14.0});
}

TEST(SparseMatrix, RejectsInvalidStorageAndMatvecSizes) {
  ExpectThrowsWithMessage<std::invalid_argument>(
      [] { (void)CsrMatrix(2, 3, {0, 2}, {0, 1}, {1.0, 2.0}); },
      "row_offset must have major dimension + 1 entries");
  ExpectThrowsWithMessage<std::invalid_argument>(
      [] { (void)CsrMatrix(2, 3, {0, 2, 1}, {0}, {1.0}); },
      "row_offset must be nondecreasing");
  ExpectThrowsWithMessage<std::out_of_range>(
      [] { (void)CsrMatrix(2, 3, {0, 1, 1}, {3}, {1.0}); },
      "col is outside matrix bounds");
  ExpectThrowsWithMessage<std::out_of_range>(
      [] { (void)CscMatrix(2, 3, {0, 1, 1, 1}, {2}, {1.0}); },
      "row is outside matrix bounds");

  const CsrMatrix csr = CooMatrix(2, 3).toCsr();
  const CscMatrix csc = CooMatrix(2, 3).toCsc();

  ExpectThrowsWithMessage<std::invalid_argument>(
      [&] { (void)csrMatvec(csr, Vector(2)); },
      "csrMatvec x size must match matrix columns");
  ExpectThrowsWithMessage<std::invalid_argument>(
      [&] { (void)cscTransposeMatvec(csc, Vector(3)); },
      "cscTransposeMatvec y size must match matrix rows");
}

}  // namespace
}  // namespace pdhg::linalg
