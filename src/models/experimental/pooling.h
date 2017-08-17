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

Expr ConvolutionInTime(std::string name, Expr x, Expr mask, int k, int n) {
  return Convolution(name, k, 1, 1, k/2, 0)(x, mask, n);
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
    int dimEmb = opt<std::vector<int>>("dim-emb")[batchIdx];
    int dimHid = opt<std::vector<int>>("dim-rnn")[batchIdx];

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

    int dimMaxLength = 1024;
    auto posFactory = embedding(graph)
                      ("prefix", prefix_ + "_Pemb")
                      ("dimVocab", dimMaxLength)
                      ("dimEmb", dimEmb);

    auto pEmb = posFactory.construct();

    float dropProb = inference_ ? 0 : opt<float>("dropout-src");
    if(dropProb) {
      int srcWords = pEmb->shape()[2];
      auto dropMask = graph->dropout(dropProb, {1, 1, srcWords});
      pEmb = dropout(pEmb, mask = dropMask);
    }

    std::vector<size_t> pIndices;
    for(int i = 0; i < dimSrcWords; ++i)
      for(int j = 0; j < dimBatch; j++)
        pIndices.push_back(i >= dimMaxLength ? 0 : i);

    auto p = reshape(rows(pEmb, pIndices), {dimBatch, dimEmb, dimSrcWords});
    auto x = (w + p) * xMask;

    int k = opt<int>("conv-width");
    std::string convType = opt<std::string>("conv-type");
    if (convType == "pooling") {
      auto c = MeanInTime(x, xMask, k);
      return New<EncoderStatePooling>(c, x, xMask, batch);
    } else if (convType == "conv") {
      int layersC = 6;
      Expr cnnC;
      if (dimEmb != dimHid) {
        auto Wup = graph->param("W_c_up", {dimEmb, dimHid}, init=inits::glorot_uniform);
        auto Bup = graph->param("b_c_up", {1, dimHid}, init=inits::zeros);
        auto cnnC = affine(x, Wup, Bup);
      } else {
        cnnC = x;
      }

      cnnC = ConvolutionInTime("cnn-c.", cnnC, xMask, k, layersC);

      if (dimEmb != dimHid) {
        auto Wdown = graph->param("W_c_down", {dimHid, dimEmb}, init=inits::glorot_uniform);
        auto Bdown = graph->param("b_c_down", {1, dimEmb}, init=inits::zeros);
        cnnC = affine(cnnC, Wdown, Bdown) * xMask;
      }

      int layersA = layersC / 2;
      auto cnnA = ConvolutionInTime("cnn-a.", x, xMask, k, layersA);
      return New<EncoderStatePooling>(cnnC, cnnA, xMask, batch);
    }
    return nullptr;
  }

    /* } else if (convType == "conv-rnn") { */
      // int layersC = 6;
      // int layersA = 3;
      // auto Wup = graph->param("W_c_up", {dimEmb, 2 * dimEmb}, init=inits::glorot_uniform);
      // auto Bup = graph->param("b_c_up", {1, 2 * dimEmb}, init=inits::zeros);

      // auto Wdown = graph->param("W_c_down", {2 * dimEmb, dimEmb}, init=inits::glorot_uniform);
      // auto Bdown = graph->param("b_c_down", {1, dimEmb}, init=inits::zeros);

      // auto cnnC = affine(x, Wup, Bup);
      // for (int i = 0; i < layersC; ++i) {
        // cnnC = ConvolutionInTime("cnn-c." + std::to_string(i), cnnC, xMask, k);
      // }

      // cnnC = affine(cnnC, Wdown, Bdown) * xMask;

      // int first = opt<int>("enc-depth");

      // auto forward = rnn::dir::forward;
      // auto backward = rnn::dir::backward;

      // using namespace keywords;
      // float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");
      // auto rnnFw = rnn::rnn(graph)
                  // ("type", opt<std::string>("enc-cell"))
                  // ("direction", forward)
                  // ("dimInput", opt<int>("dim-emb"))
                  // ("dimState", opt<int>("dim-rnn"))
                  // ("dropout", dropoutRnn)
                  // ("layer-normalization", opt<bool>("layer-normalization"))
                  // ("skip", opt<bool>("skip"));

      // for(int i = 1; i <= first; ++i) {
        // auto stacked = rnn::stacked_cell(graph);
        // for(int j = 1; j <= opt<int>("enc-cell-depth"); ++j) {
          // std::string paramPrefix = prefix_ + "_bi";
          // if(i > 1)
            // paramPrefix += "_l" + std::to_string(i);
          // if(i > 1 || j > 1)
            // paramPrefix += "_cell" + std::to_string(j);
          // stacked.push_back(rnn::cell(graph)
                            // ("prefix", paramPrefix));
        // }
        // rnnFw.push_back(stacked);
      // }

      // auto rnnBw = rnn::rnn(graph)
                  // ("type", opt<std::string>("enc-cell"))
                  // ("direction", backward)
                  // ("dimInput", opt<int>("dim-emb"))
                  // ("dimState", opt<int>("dim-rnn"))
                  // ("dropout", dropoutRnn)
                  // ("layer-normalization", opt<bool>("layer-normalization"))
                  // ("skip", opt<bool>("skip"));

      // for(int i = 1; i <= first; ++i) {
        // auto stacked = rnn::stacked_cell(graph);
        // for(int j = 1; j <= opt<int>("enc-cell-depth"); ++j) {
          // std::string paramPrefix = prefix_ + "_bi_r";
          // if(i > 1)
            // paramPrefix += "_l" + std::to_string(i);
          // if(i > 1 || j > 1)
            // paramPrefix += "_cell" + std::to_string(j);
          // stacked.push_back(rnn::cell(graph)
                            // ("prefix", paramPrefix));
        // }
        // rnnBw.push_back(stacked);
      // }

      // auto context = concatenate({rnnFw->transduce(cnnC, xMask),
                                  // rnnBw->transduce(cnnC, xMask)},
                                  // axis=1);
      // auto cnnA = x;
      // for (int i = 0; i < layersA; ++i) {
        // cnnA = ConvolutionInTime("cnn-a." + std::to_string(i), cnnA, xMask, k);
      // }
      // auto WAdown = graph->param("W_a_down", {dimEmb, 2 * dimEmb}, init=inits::glorot_uniform);
      // auto BAdown = graph->param("b_a_down", {1, 2 * dimEmb}, init=inits::zeros);

      // cnnA = affine(cnnA, WAdown, BAdown);
      // return New<EncoderStatePooling>(context, cnnA, xMask, batch);

};

typedef EncoderDecoder<EncoderPooling, DecoderS2S> PoolingModel;


}
