#pragma once

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "3rd_party/exception.h"
#include "common/definitions.h"
#include "common/shape.h"
#include "tensors/memory_piece.h"

namespace marian {

//class enum Type {
//  Float,
//  Int
//};
//
//const size_t SIZE_OF[] = { 4, 4 };

class TensorBase : public std::enable_shared_from_this<TensorBase> {
private:
  Ptr<MemoryPiece> memory_;
  Shape shape_;
  size_t device_;

  //Type type_{Type::Float};

public:
  TensorBase(Ptr<MemoryPiece> memory, Shape shape, size_t device/*, Type type = Type::Float*/)
      : memory_(memory), shape_(shape), device_(device)/*, type_(type)*/ {}

  ~TensorBase() {}

  void reset(Ptr<MemoryPiece> memory) { memory_ = memory; }

  Ptr<MemoryPiece> memory() { return memory_; }

  //Type& type() { return type_; }

  //size_t sizeOfType() { return SIZE_OF[type_] };

  Shape& shape() { return shape_; }

  template <typename T = float>
  T* data() { return (T*)memory_->data(); }

  size_t size() { return shape_.elements(); }

  template <typename T = float>
  T scalar() {
    UTIL_THROW_IF2(size() != 1, "Tensor is not a scalar");
    return get<T>(0);
  }

  size_t getDevice() { return device_; }

  template <typename T = float>
  Tensor subtensor(int offset, int size) {
    auto mem = New<MemoryPiece>(memory_->data() + sizeof(T) * offset,
                                sizeof(T) * size);
    return Tensor(new TensorBase(mem, {1, size}, device_));
  }

  void get(void* value, size_t sizeOf, size_t num, size_t offset);

  template <typename T=float>
  T get(size_t i) {
    T value;
    get(&value, sizeof(T), 1, i);
    return value;
  }

  void set(const void* value, size_t sizeOf, size_t num, size_t offset);

  template <typename T=float>
  void set(size_t i, T value) {
    set(&value, sizeof(T), 1, i);
  }

  template <typename T=float>
  void get(std::vector<T>& v) {
    get(v.data(), sizeof(T), v.size(), 0);
  }

  // @TODO: make this better
  void set(float value);

  template <typename T>
  void set(const std::vector<T>& v) {
    set(v.data(), sizeof(T), v.size(), 0);
  }

  void setSparse(const std::vector<size_t>& k, const std::vector<float>& v);

  void copyFrom(Tensor);

  template <typename T=float>
  std::string debug() {

    std::stringstream strm;
    assert(shape_.size());
    strm << shape_;
    strm << " device=" << device_;
    strm << " ptr=" << (size_t)memory_->data();
    strm << " bytes=" << memory_->size();
    strm << std::endl;

    // values
    size_t totSize = shape_.elements();
    std::vector<T> values(totSize);
    get(values);

    size_t dispCols = 5;
    strm << std::fixed << std::setprecision(8) << std::setfill(' ');
    for(size_t l = 0; l < shape()[3]; ++l) {
      for(size_t k = 0; k < shape()[2]; ++k) {
        strm << "[ ";
        if(shape()[0] > 10) {
          for(size_t i = 0; i < shape()[0] && i < dispCols; ++i) {
            if(i > 0)
              strm << std::endl << "  ";
            for(size_t j = 0; j < shape()[1] && j < dispCols; ++j) {
              strm << std::setw(12)
                   << values[i * shape().stride(0) + j * shape().stride(1)
                             + k * shape().stride(2)
                             + l * shape().stride(3)]
                   << " ";
            }
            if(shape()[1] > dispCols)
              strm << "... ";
            for(size_t j = shape()[1] - dispCols; j < shape()[1]; ++j) {
              strm << std::setw(12)
                   << values[i * shape().stride(0) + j * shape().stride(1)
                             + k * shape().stride(2)
                             + l * shape().stride(3)]
                   << " ";
            }
          }
          strm << std::endl << "  ...";
          for(size_t i = shape()[0] - dispCols; i < shape()[0]; ++i) {
            if(i > 0)
              strm << std::endl << "  ";
            for(size_t j = 0; j < shape()[1] && j < dispCols; ++j) {
              strm << std::setw(12)
                   << values[i * shape().stride(0) + j * shape().stride(1)
                             + k * shape().stride(2)
                             + l * shape().stride(3)]
                   << " ";
            }
            if(shape()[1] > dispCols)
              strm << "... ";
            for(size_t j = shape()[1] - dispCols; j < shape()[1]; ++j) {
              strm << std::setw(12)
                   << values[i * shape().stride(0) + j * shape().stride(1)
                             + k * shape().stride(2)
                             + l * shape().stride(3)]
                   << " ";
            }
          }
        } else {
          for(size_t i = 0; i < shape()[0] && i < 10; ++i) {
            if(i > 0)
              strm << std::endl << "  ";
            for(size_t j = 0; j < shape()[1] && j < dispCols; ++j) {
              strm << std::setw(12)
                   << values[i * shape().stride(0) + j * shape().stride(1)
                             + k * shape().stride(2)
                             + l * shape().stride(3)]
                   << " ";
            }
            if(shape()[1] > dispCols)
              strm << "... ";
            for(size_t j = shape()[1] - dispCols; j < shape()[1]; ++j) {
              strm << std::setw(12)
                   << values[i * shape().stride(0) + j * shape().stride(1)
                             + k * shape().stride(2)
                             + l * shape().stride(3)]
                   << " ";
            }
          }
        }
        strm << "]" << std::endl;
      }
    }
    return strm.str();
}
};

typedef std::shared_ptr<TensorBase> Tensor;

Tensor operator<<(Tensor t, const std::vector<float>& v);

Tensor operator>>(Tensor t, std::vector<float>& v);
}
