#include "linear_algebra.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <gsl/gsl>
#include <magic_enum/magic_enum.hpp>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace pdhg::linalg {
namespace {

void checkNonnegativeDimension(Index value, MatrixDimension dimension) {
  if (value < 0) {
    throw std::invalid_argument(
        std::format("{}s must be nonnegative", magic_enum::enum_name(dimension)));
  }
}

std::size_t toSize(Index value, MatrixDimension dimension) {
  checkNonnegativeDimension(value, dimension);
  return gsl::narrow<std::size_t>(value);
}

std::size_t toSize(Index value) { return gsl::narrow<std::size_t>(value); }

Index toIndex(std::size_t value) { return gsl::narrow<Index>(value); }

void checkIndex(Index index, Index limit, MatrixDimension dimension) {
  if (index < 0 || index >= limit) {
    throw std::out_of_range(
        std::format("{} is outside matrix bounds", magic_enum::enum_name(dimension)));
  }
}

void checkSameSize(const Vector& lhs, const Vector& rhs, VectorOperation operation) {
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument(std::format("{} requires equal vector sizes",
                                            magic_enum::enum_name(operation)));
  }
}

std::vector<CooEntry> sortedCombinedEntries(const std::vector<CooEntry>& entries) {
  std::vector<CooEntry> sorted = entries;
  std::sort(sorted.begin(), sorted.end(), [](const CooEntry& lhs, const CooEntry& rhs) {
    return std::pair(lhs.row, lhs.col) < std::pair(rhs.row, rhs.col);
  });

  std::vector<CooEntry> combined;
  for (const CooEntry& entry : sorted) {
    if (!combined.empty() && combined.back().row == entry.row &&
        combined.back().col == entry.col) {
      combined.back().value += entry.value;
    } else {
      combined.push_back(entry);
    }
  }

  combined.erase(
      std::remove_if(combined.begin(), combined.end(),
                     [](const CooEntry& entry) { return entry.value == Scalar{0}; }),
      combined.end());
  return combined;
}

void checkSparseStorage(Index major_dim, MatrixDimension major_dimension,
                        Index minor_dim, MatrixDimension minor_dimension,
                        const std::vector<Index>& offset, const std::vector<Index>& idx,
                        const std::vector<Scalar>& values) {
  checkNonnegativeDimension(major_dim, major_dimension);
  checkNonnegativeDimension(minor_dim, minor_dimension);
  if (offset.size() != toSize(major_dim, major_dimension) + 1) {
    throw std::invalid_argument(
        std::format("{}_offset must have major dimension + 1 entries",
                    magic_enum::enum_name(major_dimension)));
  }
  if (idx.size() != values.size()) {
    throw std::invalid_argument(std::format("{}_idx and values must have equal sizes",
                                            magic_enum::enum_name(minor_dimension)));
  }
  if (offset.empty() || offset.front() != 0) {
    throw std::invalid_argument(std::format("{}_offset must start at zero",
                                            magic_enum::enum_name(major_dimension)));
  }
  for (std::size_t i = 1; i < offset.size(); ++i) {
    if (offset[i] < offset[i - 1]) {
      throw std::invalid_argument(std::format("{}_offset must be nondecreasing",
                                              magic_enum::enum_name(major_dimension)));
    }
  }
  if (offset.back() != toIndex(values.size())) {
    throw std::invalid_argument(
        std::format("{}_offset must end at the number of values",
                    magic_enum::enum_name(major_dimension)));
  }
  for (Index index : idx) {
    checkIndex(index, minor_dim, minor_dimension);
  }
}

}  // namespace

MetalBuffer::~MetalBuffer() {
  if (buffer_ != nullptr) {
    buffer_->release();
  }
}

MetalBuffer::MetalBuffer(MetalBuffer&& other) noexcept
    : buffer_(other.buffer_), bytes_(other.bytes_) {
  other.buffer_ = nullptr;
  other.bytes_ = 0;
}

MetalBuffer& MetalBuffer::operator=(MetalBuffer&& other) noexcept {
  if (this != &other) {
    if (buffer_ != nullptr) {
      buffer_->release();
    }
    buffer_ = other.buffer_;
    bytes_ = other.bytes_;
    other.buffer_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

Vector::Vector(std::size_t size, Scalar value) : values_(size, value) {}

Vector::Vector(std::initializer_list<Scalar> values) : values_(values) {}

void Vector::fill(Scalar value) { std::fill(values_.begin(), values_.end(), value); }

void Vector::scale(Scalar alpha) {
  for (Scalar& value : values_) {
    value *= alpha;
  }
}

void Vector::axpy(Scalar alpha, const Vector& x) {
  checkSameSize(*this, x, VectorOperation::axpy);
  for (std::size_t i = 0; i < values_.size(); ++i) {
    values_[i] += alpha * x[i];
  }
}

void Vector::clamp(Scalar lower, Scalar upper) {
  if (lower > upper) {
    throw std::invalid_argument("clamp lower bound exceeds upper bound");
  }
  for (Scalar& value : values_) {
    value = std::clamp(value, lower, upper);
  }
}

Scalar Vector::dot(const Vector& other) const {
  checkSameSize(*this, other, VectorOperation::dot);
  Scalar result = 0.0;
  for (std::size_t i = 0; i < values_.size(); ++i) {
    result += values_[i] * other[i];
  }
  return result;
}

Scalar Vector::norm2() const { return std::sqrt(dot(*this)); }

Scalar Vector::minValue() const {
  if (values_.empty()) {
    throw std::invalid_argument("minValue requires a nonempty vector");
  }
  return *std::min_element(values_.begin(), values_.end());
}

Scalar Vector::maxValue() const {
  if (values_.empty()) {
    throw std::invalid_argument("maxValue requires a nonempty vector");
  }
  return *std::max_element(values_.begin(), values_.end());
}

CooMatrix::CooMatrix(Index rows, Index cols) : rows_(rows), cols_(cols) {
  checkNonnegativeDimension(rows, MatrixDimension::row);
  checkNonnegativeDimension(cols, MatrixDimension::col);
}

void CooMatrix::addEntry(Index row, Index col, Scalar value) {
  checkIndex(row, rows_, MatrixDimension::row);
  checkIndex(col, cols_, MatrixDimension::col);
  entries_.push_back({row, col, value});
}

namespace {

std::vector<Index> countNnzBeforeRows(Index rows,
                                      const std::vector<CooEntry>& entries) {
  std::vector<Index> nnz_before_row(toSize(rows, MatrixDimension::row) + 1, 0);
  for (const CooEntry& entry : entries) {
    ++nnz_before_row[toSize(entry.row) + 1];
  }
  std::inclusive_scan(nnz_before_row.begin(), nnz_before_row.end(),
                      nnz_before_row.begin());
  return nnz_before_row;
}

}  // namespace

CsrMatrix CooMatrix::toCsr() const {
  std::vector<CooEntry> entries = sortedCombinedEntries(entries_);
  std::vector<Index> col_idx;
  std::vector<Scalar> values;
  col_idx.reserve(entries.size());
  values.reserve(entries.size());

  auto row_offset = countNnzBeforeRows(rows_, entries);

  for (const CooEntry& entry : entries) {
    col_idx.push_back(entry.col);
    values.push_back(entry.value);
  }

  return CsrMatrix(rows_, cols_, std::move(row_offset), std::move(col_idx),
                   std::move(values));
}

CscMatrix CooMatrix::toCsc() const {
  std::vector<CooEntry> entries = sortedCombinedEntries(entries_);
  std::sort(entries.begin(), entries.end(),
            [](const CooEntry& lhs, const CooEntry& rhs) {
              return std::pair(lhs.col, lhs.row) < std::pair(rhs.col, rhs.row);
            });

  std::vector<Index> col_offset(toSize(cols_, MatrixDimension::col) + 1, 0);
  std::vector<Index> row_idx;
  std::vector<Scalar> values;
  row_idx.reserve(entries.size());
  values.reserve(entries.size());

  for (const CooEntry& entry : entries) {
    ++col_offset[toSize(entry.col) + 1];
  }
  std::partial_sum(col_offset.begin(), col_offset.end(), col_offset.begin());

  for (const CooEntry& entry : entries) {
    row_idx.push_back(entry.row);
    values.push_back(entry.value);
  }

  return CscMatrix(rows_, cols_, std::move(col_offset), std::move(row_idx),
                   std::move(values));
}

CsrMatrix::CsrMatrix(Index rows, Index cols, std::vector<Index> row_offset,
                     std::vector<Index> col_idx, std::vector<Scalar> values)
    : rows_(rows),
      cols_(cols),
      row_offset_(std::move(row_offset)),
      col_idx_(std::move(col_idx)),
      values_(std::move(values)) {
  checkSparseStorage(rows_, MatrixDimension::row, cols_, MatrixDimension::col,
                     row_offset_, col_idx_, values_);
}

CscMatrix::CscMatrix(Index rows, Index cols, std::vector<Index> col_offset,
                     std::vector<Index> row_idx, std::vector<Scalar> values)
    : rows_(rows),
      cols_(cols),
      col_offset_(std::move(col_offset)),
      row_idx_(std::move(row_idx)),
      values_(std::move(values)) {
  checkSparseStorage(cols_, MatrixDimension::col, rows_, MatrixDimension::row,
                     col_offset_, row_idx_, values_);
}

void csrMatvec(const CsrMatrix& matrix, const Vector& x, Vector& y) {
  if (x.size() != toSize(matrix.cols(), MatrixDimension::col)) {
    throw std::invalid_argument("csrMatvec x size must match matrix columns");
  }
  if (y.size() != toSize(matrix.rows(), MatrixDimension::row)) {
    throw std::invalid_argument("csrMatvec y size must match matrix rows");
  }

  const auto rows = toSize(matrix.rows(), MatrixDimension::row);
  for (std::size_t row = 0; row < rows; ++row) {
    Scalar sum = 0.0;
    for (auto offset = toSize(matrix.rowOffset()[row]);
         offset < toSize(matrix.rowOffset()[row + 1]); ++offset) {
      sum += matrix.values()[offset] * x[toSize(matrix.colIdx()[offset])];
    }
    y[row] = sum;
  }
}

void cscTransposeMatvec(const CscMatrix& matrix, const Vector& y, Vector& x) {
  if (y.size() != toSize(matrix.rows(), MatrixDimension::row)) {
    throw std::invalid_argument("cscTransposeMatvec y size must match matrix rows");
  }
  if (x.size() != toSize(matrix.cols(), MatrixDimension::col)) {
    throw std::invalid_argument("cscTransposeMatvec x size must match matrix columns");
  }

  const auto cols = toSize(matrix.cols(), MatrixDimension::col);
  for (std::size_t col = 0; col < cols; ++col) {
    Scalar sum = 0.0;
    for (auto offset = toSize(matrix.colOffset()[col]);
         offset < toSize(matrix.colOffset()[col + 1]); ++offset) {
      sum += matrix.values()[offset] * y[toSize(matrix.rowIdx()[offset])];
    }
    x[col] = sum;
  }
}

Vector csrMatvec(const CsrMatrix& matrix, const Vector& x) {
  Vector y(toSize(matrix.rows(), MatrixDimension::row));
  csrMatvec(matrix, x, y);
  return y;
}

Vector cscTransposeMatvec(const CscMatrix& matrix, const Vector& y) {
  Vector x(toSize(matrix.cols(), MatrixDimension::col));
  cscTransposeMatvec(matrix, y, x);
  return x;
}

}  // namespace pdhg::linalg
