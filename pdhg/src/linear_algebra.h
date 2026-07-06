#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace pdhg::linalg {

enum class MatrixDimension : std::uint8_t {
  row,
  col,
};

enum class VectorOperation : std::uint8_t {
  axpy,
  dot,
};

using Scalar = double;
using Index = std::int32_t;

class MetalBuffer {
 public:
  MetalBuffer() = default;

  MetalBuffer(MTL::Device* device, std::size_t bytes)
      : buffer_(device->newBuffer(bytes, MTL::ResourceStorageModeShared)),
        bytes_(bytes) {}

  ~MetalBuffer();

  MetalBuffer(const MetalBuffer&) = delete;
  MetalBuffer& operator=(const MetalBuffer&) = delete;

  MetalBuffer(MetalBuffer&& other) noexcept;
  MetalBuffer& operator=(MetalBuffer&& other) noexcept;

  [[nodiscard]] MTL::Buffer* get() const { return buffer_; }
  [[nodiscard]] std::size_t bytes() const { return bytes_; }
  [[nodiscard]] void* contents() const { return buffer_->contents(); }

 private:
  // Metal objects use retain/release ownership, not C++ delete ownership.
  MTL::Buffer* buffer_ = nullptr;
  std::size_t bytes_ = 0;
};

class Vector {
 public:
  Vector() = default;
  explicit Vector(std::size_t size, Scalar value = 0.0);
  Vector(std::initializer_list<Scalar> values);

  [[nodiscard]] std::size_t size() const { return values_.size(); }
  [[nodiscard]] bool empty() const { return values_.empty(); }
  [[nodiscard]] Scalar* data() { return values_.data(); }
  [[nodiscard]] const Scalar* data() const { return values_.data(); }
  [[nodiscard]] const std::vector<Scalar>& values() const { return values_; }

  [[nodiscard]] Scalar& operator[](std::size_t index) { return values_[index]; }
  [[nodiscard]] Scalar operator[](std::size_t index) const { return values_[index]; }

  void fill(Scalar value);
  void scale(Scalar alpha);
  void axpy(Scalar alpha, const Vector& x);
  void clamp(Scalar lower, Scalar upper);

  [[nodiscard]] Scalar dot(const Vector& other) const;
  [[nodiscard]] Scalar norm2() const;
  [[nodiscard]] Scalar minValue() const;
  [[nodiscard]] Scalar maxValue() const;

 private:
  std::vector<Scalar> values_;
};

class CsrMatrix {
 public:
  CsrMatrix() = default;
  CsrMatrix(Index rows, Index cols, std::vector<Index> row_offset,
            std::vector<Index> col_idx, std::vector<Scalar> values);

  [[nodiscard]] Index rows() const { return rows_; }
  [[nodiscard]] Index cols() const { return cols_; }
  [[nodiscard]] std::size_t nnz() const { return values_.size(); }
  [[nodiscard]] const std::vector<Index>& rowOffset() const { return row_offset_; }
  [[nodiscard]] const std::vector<Index>& colIdx() const { return col_idx_; }
  [[nodiscard]] const std::vector<Scalar>& values() const { return values_; }

 private:
  Index rows_ = 0;
  Index cols_ = 0;
  std::vector<Index> row_offset_;
  std::vector<Index> col_idx_;
  std::vector<Scalar> values_;
};

class CscMatrix {
 public:
  CscMatrix() = default;
  CscMatrix(Index rows, Index cols, std::vector<Index> col_offset,
            std::vector<Index> row_idx, std::vector<Scalar> values);

  [[nodiscard]] Index rows() const { return rows_; }
  [[nodiscard]] Index cols() const { return cols_; }
  [[nodiscard]] std::size_t nnz() const { return values_.size(); }
  [[nodiscard]] const std::vector<Index>& colOffset() const { return col_offset_; }
  [[nodiscard]] const std::vector<Index>& rowIdx() const { return row_idx_; }
  [[nodiscard]] const std::vector<Scalar>& values() const { return values_; }

 private:
  Index rows_ = 0;
  Index cols_ = 0;
  std::vector<Index> col_offset_;
  std::vector<Index> row_idx_;
  std::vector<Scalar> values_;
};

struct CooEntry {
  Index row = 0;
  Index col = 0;
  Scalar value = 0.0;
};

class CooMatrix {
 public:
  CooMatrix(Index rows, Index cols);

  [[nodiscard]] Index rows() const { return rows_; }
  [[nodiscard]] Index cols() const { return cols_; }
  [[nodiscard]] std::size_t nnz() const { return entries_.size(); }
  [[nodiscard]] const std::vector<CooEntry>& entries() const { return entries_; }

  void addEntry(Index row, Index col, Scalar value);

  [[nodiscard]] CsrMatrix toCsr() const;
  [[nodiscard]] CscMatrix toCsc() const;

 private:
  Index rows_ = 0;
  Index cols_ = 0;
  std::vector<CooEntry> entries_;
};

struct LpProblem {
  CsrMatrix constraints_by_row;
  CscMatrix constraints_by_col;
  Vector objective;
  Vector lower_bounds;
  Vector upper_bounds;
  Vector rhs;
};

void csrMatvec(const CsrMatrix& matrix, const Vector& x, Vector& y);
void cscTransposeMatvec(const CscMatrix& matrix, const Vector& y, Vector& x);

[[nodiscard]] Vector csrMatvec(const CsrMatrix& matrix, const Vector& x);
[[nodiscard]] Vector cscTransposeMatvec(const CscMatrix& matrix, const Vector& y);

}  // namespace pdhg::linalg
