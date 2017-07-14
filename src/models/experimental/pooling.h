#pragma once

#include "models/encdec.h"
#include "models/s2s.h"

namespace marian {

class EncoderStatePooling : public EncoderState {
private:
  Expr context_;
  Expr attended_;
  Expr mask_;
  Ptr<data::CorpusBatch> batch_;

public:
  EncoderStatePooling(Expr context, Expr attended, Expr mask, Ptr<data::CorpusBatch> batch)
      : EncoderState(context, mask, batch),
        attended_(attended)
  {}

  virtual Expr getAttended() { return attended_; }
};

Expr MeanInTime(Ptr<ExpressionGraph> graph, Expr x, int k) {
    using namespace keywords;
    int dimBatch = x->shape()[0];
    int dimInput = x->shape()[1];
    int dimSrcWords = x->shape()[2];

    auto padding = graph->zeros(shape={dimBatch, dimInput, k / 2});
    auto xpad = concatenate({padding, x, padding}, axis=2);

    std::vector<Expr> means;
    for(int i = 0; i < dimSrcWords; ++i) {
      std::vector<Expr> preAvg;
      for(int j = 0; j < k; ++j)
        preAvg.push_back(step(xpad, i + j));

      means.push_back(mean(concatenate(preAvg, axis=2), axis=2));
    }
    return tanh(concatenate(means, axis=2) + x);
}

Expr ConvolutionInTime(Ptr<ExpressionGraph> graph, Expr x,
                       int k, std::string name) {
    using namespace keywords;
    int dimBatch = x->shape()[0];
    int dimInput = x->shape()[1];
    int dimSrcWords = x->shape()[2];

    auto padding = graph->zeros(shape={dimBatch, dimInput, k / 2});
    auto xpad = concatenate({padding, x, padding}, axis=2);

    float scale = 1.f / sqrtf(k * dimInput);
    auto K = graph->param(name, {1, dimInput, k}, keywords::init=inits::uniform(scale));
    auto B = graph->param(name + "_b", {1, dimInput}, init=inits::zeros);

    std::vector<Expr> filters;
    for(int i = 0; i < dimSrcWords; ++i) {
      std::vector<Expr> preAvg;
      for(int j = 0; j < k; ++j)
        preAvg.push_back(step(xpad, i + j));

      filters.push_back(sum(concatenate(preAvg, axis=2) * K, axis=2));
    }
    return tanh(concatenate(filters, axis=2), B, x);
}


class EncoderPooling : public EncoderBase {
public:
  template <class... Args>
  EncoderPooling(Ptr<Config> options, Args... args)
      : EncoderBase(options, args...) {}

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch,
                          size_t batchIdx) {
    using namespace keywords;

    int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIdx];
    int dimEmb = opt<int>("dim-emb");

    auto embFactory = embedding(graph)
                      ("prefix", prefix_ + "_Wemb")
                      ("dimVocab", dimVoc)
                      ("dimEmb", dimEmb);

    if(options_->has("embedding-fix-src"))
      embFactory
        ("fixed", opt<bool>("embedding-fix-src"));

    if(options_->has("embedding-vectors")) {
      auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
      embFactory
        ("embFile", embFiles[batchIdx])
        ("normalization", opt<bool>("embedding-normalization"));
    }

    auto xEmb = embFactory.construct();

    Expr w, xMask;
    std::tie(w, xMask) = lookup(xEmb, batch, batchIdx);

    int dimBatch = w->shape()[0];
    int dimSrcWords = w->shape()[2];

    int dimMaxLength = options_->get<size_t>("max-length") + 1;
    auto pEmb = embedding(graph)
                ("prefix", prefix_ + "_Pemb")
                ("dimVocab", dimMaxLength)
                ("vimEmb", dimEmb)
                .construct();

    std::vector<size_t> pIndices;
    for (int i = 0; i < dimSrcWords; ++i)
      for (int j = 0; j < dimBatch; ++j)
        pIndices.push_back(i);

    auto p = reshape(rows(pEmb, pIndices), {dimBatch, dimEmb, dimSrcWords});
    auto x = w + p;

    int k = 3;
    int layersC = 6;
    int layersA = 3;

    auto Wup = graph->param("W_c_up", {dimEmb, 2 * dimEmb}, init=inits::glorot_uniform);
    auto Bup = graph->param("b_c_up", {1, 2 * dimEmb}, init=inits::zeros);

    auto Wdown = graph->param("W_c_down", {2 * dimEmb, dimEmb}, init=inits::glorot_uniform);
    auto Bdown = graph->param("b_c_down", {1, dimEmb}, init=inits::zeros);

    auto cnnC = affine(x, Wup, Bup);
    for (int i = 0; i < layersC; ++i) {
      cnnC = ConvolutionInTime(graph, cnnC, k, "cnn-c." + std::to_string(i));
    }
    cnnC = affine(cnnC, Wdown, Bdown);

    auto cnnA = x;
    for (int i = 0; i < layersA; ++i) {
      cnnA = ConvolutionInTime(graph, cnnA, k, "cnn-a." + std::to_string(i));
    }

    return New<EncoderStatePooling>(cnnC, cnnA + x, xMask, batch);
  }
};

typedef EncoderDecoder<EncoderPooling, DecoderS2S> PoolingModel;


}
