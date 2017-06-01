#pragma once

#include <string>

#include "layers/generic.h"

namespace marian {

class Convolution : public Layer {
  public:
    Convolution(const std::string& name, int kernelHeight = 3, int kernelNum = 1, int depth = 1)
      : Layer(name), depth_(depth), kernelHeight_(kernelHeight), kernelNum_(kernelNum) {
    }

    Expr operator()(Expr x, Expr mask) {
      params_ = {};

      auto graph = x->graph();

      std::vector<size_t> newIndeces;
      int batchDim = x->shape()[0];
      int sentenceDim = x->shape()[2];

      for (int b = 0; b < batchDim; ++b) {
        for (int t = 0; t < sentenceDim; ++t) {
          newIndeces.push_back((t * batchDim) + b);
        }
      }

      auto masked = reshape(x * mask, {batchDim * sentenceDim, x->shape()[1], 1, x->shape()[3]});
      auto shuffled_X = reshape(rows(masked, newIndeces), {batchDim, 1, sentenceDim, x->shape()[1]});
      auto shuffled_mask = reshape(rows(mask, newIndeces), {batchDim, 1, sentenceDim, mask->shape()[1]});

      // debug(shuffled_X, "SX");
      // debug(shuffled_mask, "SMASK");

      Expr* previousInput = &shuffled_X;

      for (int layerIdx = 0; layerIdx < depth_; ++layerIdx) {
        std::string kernel_name = name_ + "kernel_" + std::to_string(layerIdx);
        auto kernel = graph->param(kernel_name,  {kernelNum_, kernelHeight_},
                                   keywords::init=inits::glorot_uniform);
                                   // keywords::init=inits::ones);
        params_.push_back(kernel);
        // debug(kernel, kernel_name);

        auto input = *previousInput * shuffled_mask;
        auto output = tanh(input + convolution(input, kernel));
        // debug(output, "OUTPUT");
        previousInput = &output;
      }
      auto reshapedOutput = reshape(*previousInput * shuffled_mask, {batchDim * sentenceDim, x->shape()[1],
                                                     1, x->shape()[3]});

      // debug(reshapedOutput, "RESHAPED OUTPUT");
      newIndeces.clear();
      for (int t = 0; t < sentenceDim; ++t) {
        for (int b = 0; b < batchDim; ++b) {
          newIndeces.push_back(b * sentenceDim + t);
        }
      }

      auto reshaped = reshape(rows(reshapedOutput, newIndeces), x->shape());
      // debug(reshaped, "RESHAPED");
      return reshaped * mask;
    }

  protected:
    const int depth_;
    const int kernelHeight_;
    const int kernelNum_;
};

class Pooling : public Layer {
  public:
    Pooling(const std::string& name)
      : Layer(name) {
    }

    Expr operator()(Expr x, Expr xMask) {
      params_ = {};

      std::vector<size_t> newIndeces;
      int batchDim = x->shape()[0];
      int sentenceDim = x->shape()[2];

      for (int b = 0; b < batchDim; ++b) {
        for (int t = 0; t < sentenceDim; ++t) {
          newIndeces.push_back((t * batchDim) + b);
        }
      }

      auto masked = reshape(x * xMask, {batchDim * sentenceDim, x->shape()[1], 1, x->shape()[3]});
      auto newX = reshape(rows(masked, newIndeces), {batchDim, x->shape()[1], sentenceDim, 1});

      auto pooled = reshape(avg_pooling(newX), {batchDim * sentenceDim, x->shape()[1], 1, x->shape()[3]});

      newIndeces.clear();
      for (int t = 0; t < sentenceDim; ++t) {
        for (int b = 0; b < batchDim; ++b) {
          newIndeces.push_back(b * sentenceDim + t);
        }
      }

      auto reshaped = reshape(rows(pooled, newIndeces), x->shape());
      return reshaped * xMask;
    }
};

}
