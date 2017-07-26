#pragma once

#include "models/encdec.h"
#include "models/s2s.h"
#include "rnn/attention.h"

namespace marian {

class EncoderT2T : public EncoderBase {
public:
  template <class... Args>
  EncoderT2T(Ptr<Config> options, Args... args)
      : EncoderBase(options, args...) {}

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch,
                          size_t encoderIndex) {
    using namespace keywords;

    // create source embeddings
    int dimVoc = opt<std::vector<int>>("dim-vocabs")[encoderIndex];
    int dimEmb = opt<int>("dim-emb");

    auto embeddings = embedding(graph)
                      ("prefix", prefix_ + "_Wemb")
                      ("dimVocab", dimVoc)
                      ("dimEmb", dimEmb)
                      .construct();

    Expr w, batchMask;
    std::tie(w, batchMask)
      = EncoderBase::lookup(embeddings, batch, encoderIndex);

    int dimBatch = w->shape()[0];
    int dimSrcWords = w->shape()[2];

    int dimMaxLength = 50;
    auto pEmb = embedding(graph)
                ("prefix", prefix_ + "_Pemb")
                ("dimVocab", dimMaxLength)
                ("dimEmb", dimEmb)
                .construct();

    std::vector<size_t> pIndices;
    for(int i = 0; i < dimSrcWords; ++i)
      for(int j = 0; j < dimBatch; j++)
        pIndices.push_back(i);

    auto p = reshape(rows(pEmb, pIndices), {dimBatch, dimEmb, dimSrcWords});
    auto x = w + p;

    int layers = 3;
    Expr layerOut = x;
    for(int i = 0; i < layers; i++) {

      auto opt = New<Options>();
      opt->set("dimState", dimEmb);
      opt->set("dropout", 0);
      opt->set("layer-normalization", false);
      opt->set("prefix", prefix_ + "_att" + std::to_string(i));
      auto att = New<rnn::GlobalAttention>(graph, opt, layerOut, layerOut, batchMask);

      std::vector<Expr> attsteps;
      for(int j = 0; j < dimSrcWords; ++j)
        attsteps.push_back(att->apply(step(layerOut, j)));

      auto layerAtt = concatenate(attsteps, axis=2);
      auto gamma1 = graph->param(prefix_ + "_gamma1" + std::to_string(i),
                                 {1, dimEmb},
                                 keywords::init = inits::from_value(1.f));
      layerAtt = layer_norm(layerAtt + layerOut, gamma1);

      // FFN
      auto Wup = graph->param(prefix_ + "_W1" + std::to_string(i),
                              {dimEmb, 2 * dimEmb}, init=inits::glorot_uniform);
      auto Bup = graph->param(prefix_ + "_b1" + std::to_string(i),
                              {1, 2 * dimEmb}, init=inits::zeros);

      auto Wdown = graph->param(prefix_ + "_W2" + std::to_string(i),
                                {2 * dimEmb, dimEmb}, init=inits::glorot_uniform);
      auto Bdown = graph->param(prefix_ + "_b2" + std::to_string(i),
                                {1, dimEmb}, init=inits::zeros);

      layerOut = affine(relu(affine(layerAtt, Wup, Bup)), Wdown, Bdown);

      auto gamma2 = graph->param(prefix_ + "_gamma2" + std::to_string(i),
                                 {1, dimEmb},
                                 keywords::init = inits::from_value(1.f));
      layerOut = layer_norm(layerOut + layerAtt, gamma2);
    }

    return New<EncoderState>(layerOut, batchMask, batch);
  }
};

typedef EncoderDecoder<EncoderT2T, DecoderS2S> PoolingT2T;


}
