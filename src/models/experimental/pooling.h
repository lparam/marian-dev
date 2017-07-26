#pragma once

#include "models/encdec.h"
#include "models/s2s.h"
#include "layers/convolution.h"

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

Expr MeanInTime(Expr x, Expr mask, int k) {
  return tanh(Pooling("Pooling", "avg_pooling", k, 1, k/2, 0)(x, mask) + x);
}

Expr ConvolutionInTime(std::string name, Expr x, Expr mask, int k) {
  return tanh(Convolution(name, k, 1, 1, k/2, 0)(x, mask) + x);
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
    std::tie(w, xMask) = EncoderBase::lookup(xEmb, batch, batchIdx);

    int dimBatch = w->shape()[0];
    int dimSrcWords = w->shape()[2];

    int dimMaxLength = 50;
    auto posFactory = embedding(graph)
                      ("prefix", prefix_ + "_Pemb")
                      ("dimVocab", dimMaxLength)
                      ("dimEmb", dimEmb);

    auto pEmb = posFactory.construct();

    std::vector<size_t> pIndices;
    for(int i = 0; i < dimSrcWords; ++i)
      for(int j = 0; j < dimBatch; j++)
        pIndices.push_back(i >= dimMaxLength ? 0 : i);

    auto p = reshape(rows(pEmb, pIndices), {dimBatch, dimEmb, dimSrcWords});
    auto x = (w + p) * xMask;

    int k = 3;
    // auto c = MeanInTime(x, xMask, k);
    // return New<EncoderStatePooling>(c, x, xMask, batch);

    int layersC = 6;
    int layersA = 3;
    auto Wup = graph->param("W_c_up", {dimEmb, 2 * dimEmb}, init=inits::glorot_uniform);
    auto Bup = graph->param("b_c_up", {1, 2 * dimEmb}, init=inits::zeros);

    auto Wdown = graph->param("W_c_down", {2 * dimEmb, dimEmb}, init=inits::glorot_uniform);
    auto Bdown = graph->param("b_c_down", {1, dimEmb}, init=inits::zeros);

    auto cnnC = affine(x, Wup, Bup);
    for (int i = 0; i < layersC; ++i) {
      cnnC = ConvolutionInTime("cnn-c." + std::to_string(i), cnnC, xMask, k);
    }
    cnnC = affine(cnnC, Wdown, Bdown) * xMask;

    auto cnnA = x;
    for (int i = 0; i < layersA; ++i) {
      cnnA = ConvolutionInTime("cnn-a." + std::to_string(i), cnnA, xMask, k);
    }
    return New<EncoderStatePooling>(cnnC, cnnA + x, xMask, batch);
  }
};

typedef EncoderDecoder<EncoderPooling, DecoderS2S> PoolingModel;


}
