#pragma once

#include <string>

#include "layers/generic.h"

namespace marian {

class Convolution {
  public:
    Convolution(
        const std::string& name,
        int kernelHeight = 3,
        int kernelWidth = 3,
        int kernelNum = 1,
        int paddingHeight = 0,
        int paddingWidth = 0,
        int strideHeight = 1,
        int strideWidth = 1)
      : name_(name),
        kernelHeight_(kernelHeight),
        kernelWidth_(kernelWidth),
        kernelNum_(kernelNum),
        strideHeight_(strideHeight),
        strideWidth_(strideWidth),
        paddingHeight_(paddingHeight),
        paddingWidth_(paddingWidth)
    {
      // std::cerr << "creating conv" << __LINE__ << std::endl;
    }

    Expr convert2NCHW(Expr x) {
      std::vector<size_t> newIndeces;
      int batchDim = x->shape()[0];
      int sentenceDim = x->shape()[2];

      for (int b = 0; b < batchDim; ++b) {
        for (int t = 0; t < sentenceDim; ++t) {
          newIndeces.push_back((t * batchDim) + b);
        }
      }

      Shape shape({batchDim, 1, sentenceDim, x->shape()[1]});
      return  reshape(rows(x, newIndeces), shape);
    }

    Expr convert2Marian(Expr x, Expr originalX) {
      std::vector<size_t> newIndeces;
      int batchDim = x->shape()[0];
      int sentenceDim = x->shape()[2];

      auto pooled = reshape(x, {batchDim * sentenceDim, x->shape()[3], 1, x->shape()[1]});

      for (int t = 0; t < sentenceDim; ++t) {
        for (int b = 0; b < batchDim; ++b) {
          newIndeces.push_back(b * sentenceDim + t);
        }
      }

      return reshape(rows(pooled, newIndeces), originalX->shape());
    }

    Expr operator()(Expr x) {
      params_.clear();
      auto graph = x->graph();

      int layerIn = x->shape()[1];

      auto kernel = graph->param(name_,
          {layerIn, kernelNum_, kernelHeight_, kernelWidth_},
          keywords::init=inits::glorot_uniform);
      auto bias = graph->param(name_ + "_bias",  {1, kernelNum_, 1, 1},
                               keywords::init=inits::zeros);
      params_.push_back(kernel);
      params_.push_back(bias);

      auto output = convolution(x, kernel, bias,
                                paddingHeight_, paddingWidth_,
                                strideHeight_, strideWidth_);

      return output;
    }

    Expr operator()(Expr x, Expr mask, int n) {
      params_ = {};
      auto graph = x->graph();

      auto masked = x * mask;
      auto xNCHW = convert2NCHW(masked);
      auto maskNCHW = convert2NCHW(mask);

      int layerIn = xNCHW->shape()[1];

      Expr input = xNCHW;
      for (int i = 0; i < n; ++i) {
        auto kernel = graph->param(name_ + std::to_string(i),
            {layerIn, kernelNum_, kernelHeight_, kernelWidth_},
            keywords::init=inits::glorot_uniform);
        auto bias = graph->param(name_ + std::to_string(i) + "_bias",  {1, kernelNum_, 1, 1},
                                keywords::init=inits::zeros);

        auto output = convolution(input, kernel, bias,
            paddingHeight_, paddingWidth_,
            strideHeight_, strideWidth_);
        input = tanh(input + output) * maskNCHW;
      }

      return convert2Marian(input, x);
    }

  private:
    std::vector<Expr> params_;
    std::string name_;

  protected:
    int depth_;
    int kernelHeight_;
    int kernelWidth_;
    int kernelNum_;
    int strideHeight_;
    int strideWidth_;
    int paddingHeight_;
    int paddingWidth_;
};

class Pooling {
public:
  Pooling(
      const std::string name,
      const std::string type,
      int height = 1,
      int width = 1,
      int paddingHeight = 0,
      int paddingWidth = 0,
      int strideHeight = 1,
      int strideWidth = 1)
    : name_(name),
      type_(type),
      height_(height),
      width_(width),
      paddingHeight_(paddingHeight),
      paddingWidth_(paddingWidth),
      strideHeight_(strideHeight),
      strideWidth_(strideWidth)
  {
  }

    Expr operator()(Expr x) {
      params_ = {};
      if (type_ == "max_pooling") {
        return max_pooling(x, height_, width_, paddingHeight_, paddingWidth_,
                           strideHeight_, strideWidth_);
      } else if (type_ == "avg_pooling") {
        return avg_pooling(x, height_, width_, paddingHeight_, paddingWidth_,
                           strideHeight_, strideWidth_);
      }
      return nullptr;
    }

    Expr convert2NCHW(Expr x) {
      std::vector<size_t> newIndeces;
      int batchDim = x->shape()[0];
      int sentenceDim = x->shape()[2];

      for (int b = 0; b < batchDim; ++b) {
        for (int t = 0; t < sentenceDim; ++t) {
          newIndeces.push_back((t * batchDim) + b);
        }
      }

      Shape shape({batchDim, 1, sentenceDim, x->shape()[1]});
      return  reshape(rows(x, newIndeces), shape);
    }

    Expr convert2Marian(Expr x, Expr originalX) {
      std::vector<size_t> newIndeces;
      int batchDim = x->shape()[0];
      int sentenceDim = x->shape()[2];

      auto pooled = reshape(x, {batchDim * sentenceDim, x->shape()[3], 1, x->shape()[1]});

      for (int t = 0; t < sentenceDim; ++t) {
        for (int b = 0; b < batchDim; ++b) {
          newIndeces.push_back(b * sentenceDim + t);
        }
      }

      return reshape(rows(pooled, newIndeces), originalX->shape());
    }

    Expr operator()(Expr x, Expr mask) {
      params_ = {};

      auto masked = x * mask;

      auto xNCHW = convert2NCHW(masked);

      Expr output;
      if (type_ == "max_pooling") {
        output = max_pooling(xNCHW, height_, width_,
                             paddingHeight_, paddingWidth_,
                             strideHeight_, strideWidth_);
      } else if (type_ == "avg_pooling") {
        output = avg_pooling(xNCHW, height_, width_,
                             paddingHeight_, paddingWidth_,
                             strideHeight_, strideWidth_);
      }
      // debug(output, "output");

      return convert2Marian(output, x) * mask;
    }

  private:
    std::vector<Expr> params_;
    std::string name_;
    std::string type_;

  protected:
    int height_;
    int width_;
    int strideHeight_;
    int strideWidth_;
    int paddingHeight_;
    int paddingWidth_;

};

}
