
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "kernels/cuda_helpers.h"
#include "kernels/tensor_operators.h"
#include "tensors/tensor.h"

namespace marian {

template <typename T>
__global__ void gFill(T *d_in, int size, T val) {
  for(int bid = 0; bid < size; bid += blockDim.x * gridDim.x) {
    int index = bid + threadIdx.x + blockDim.x * blockIdx.x;
    if(index < size) {
      d_in[index] = val;
    }
  }
}

void TensorBase::get(void* value, size_t sizeOf, size_t num, size_t offset) {
  cudaSetDevice(device_);
  CUDA_CHECK(
      cudaMemcpy(value,
                 memory_->data() + offset * sizeOf,
                 num * sizeOf,
                 cudaMemcpyDeviceToHost));
  cudaStreamSynchronize(0);
}

void TensorBase::set(const void* value, size_t sizeOf, size_t num, size_t offset) {
  cudaSetDevice(device_);
  CUDA_CHECK(
      cudaMemcpy(memory_->data() + offset * sizeOf,
                 value,
                 num * sizeOf,
                 cudaMemcpyHostToDevice));
  cudaStreamSynchronize(0);
}

void TensorBase::set(float value) {
  cudaSetDevice(device_);
  int threads = std::min(512, (int)size());
  int blocks = (size() / threads) + (size() % threads != 0);
  gFill<<<blocks, threads>>>(data<float>(), size(), value);
  cudaStreamSynchronize(0);
}

void TensorBase::setSparse(const std::vector<size_t> &k,
                           const std::vector<float> &v) {
  cudaSetDevice(device_);
  SetSparse(data(), k, v);
  cudaStreamSynchronize(0);
}

void TensorBase::copyFrom(Tensor in) {
  cudaSetDevice(device_);
  CUDA_CHECK(cudaMemcpy(
      memory()->data(), in->memory()->data(), in->memory()->size(), cudaMemcpyDefault));
  cudaStreamSynchronize(0);
}

Tensor operator<<(Tensor t, const std::vector<float> &v) {
  t->set(v);
  return t;
}

Tensor operator>>(Tensor t, std::vector<float> &v) {
  t->get(v);
  return t;
}
}
