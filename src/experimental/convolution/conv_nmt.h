#pragma once

#include "rnn/attention.h"
#include "rnn/rnn.h"

#include "layers/convolution.h"
#include "models/s2s.h"

namespace marian {

class CNNEncoderState : public EncoderState {
protected:
  Expr attended_;

public:
  CNNEncoderState(Expr context, Expr attended, Expr mask, Ptr<data::CorpusBatch> batch)
      : EncoderState(context, mask, batch),
        attended_(attended)
  {}

  virtual Expr getAttended() { return attended_; }
};


template<typename T>
std::vector<T> Convert2NCHW(const std::vector<T>& indices,
                            const Shape& shape) {
  std::vector<T> nchwIndices;
  int batchDim = shape[0];
  int sentenceDim = shape[2];

  for (int b = 0; b < batchDim; ++b) {
    for (int t = 0; t < sentenceDim; ++t) {
      nchwIndices.push_back(indices[(t * batchDim) + b]);
    }
  }
  return nchwIndices;
}


class PoolingEncoder : public EncoderBase {
public:
  Expr getContext(Ptr<ExpressionGraph> graph,
                  Expr embeddings, Expr mask) {
    using namespace keywords;

    return embeddings * mask;
  }

  Expr buildEmbeddings(Ptr<ExpressionGraph> graph,
                       size_t encoderIndex) {
    int dimVoc = opt<std::vector<int>>("dim-vocabs")[encoderIndex];
    int dimEmb = opt<int>("dim-emb");

    auto embFactory = embedding(graph)
                      ("prefix", prefix_ + "_Wemb")
                      ("dimVocab", dimVoc)
                      ("dimEmb", dimEmb);

    if (options_->has("embedding-fix-src")) {
      embFactory
        ("fixed", opt<bool>("embedding-fix-src"));
    }

    if(options_->has("embedding-vectors")) {
      auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
      embFactory
        ("embFile", embFiles[encoderIndex])
        ("normalization", opt<bool>("embedding-normalization"));
    }

    return embFactory.construct();
  }

  Expr buildPositionalEmbeddings(
      Ptr<ExpressionGraph> graph,
      size_t encoderIndex) {
    int dimEmb = opt<int>("dim-emb");
    int maxSrcLength = opt<int>("max-length");

    auto embFactory = embedding(graph)
                      ("prefix", prefix_ + "_Wemb")
                      ("dimVocab", maxSrcLength)
                      ("dimEmb", dimEmb);

    if (options_->has("embedding-fix-src")) {
      embFactory
        ("fixed", opt<bool>("embedding-fix-src"));
    }

    if(options_->has("embedding-vectors")) {
      auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
      embFactory
        ("embFile", embFiles[encoderIndex])
        ("normalization", opt<bool>("embedding-normalization"));
    }

    return embFactory.construct();
  }

  std::tuple<Expr, Expr> lookupWithPosition(
      Expr wordEmbeddings,
      Expr posEmbeddings,
      Ptr<data::CorpusBatch> batch,
      size_t index) {
    using namespace keywords;

    auto& wordIndeces = batch->at(index)->indices();
    auto& mask = batch->at(index)->mask();

    std::vector<size_t> posIndeces;

    for(size_t iPos = 0; iPos < batch->at(index)->batchWidth(); ++iPos) {
      for(size_t i = 0; i < batch->at(index)->batchSize(); ++i) {
        if(iPos < (size_t)posEmbeddings->shape()[0]) {
          posIndeces.push_back(iPos);
        } else {
          posIndeces.push_back(posEmbeddings->shape()[0] - 1);
        }
      }
    }

    int batchSize = batch->size();
    int dimEmb = wordEmbeddings->shape()[1];
    int batchLength = batch->at(index)->batchWidth();

    auto graph = wordEmbeddings->graph();

    auto xWord
        = reshape(rows(wordEmbeddings, wordIndeces), {batchSize, dimEmb, batchLength});
    auto xPos
        = reshape(rows(posEmbeddings, posIndeces), {batchSize, dimEmb, batchLength});
    auto x = xWord + xPos;
    auto xMask = graph->constant({batchSize, 1, batchLength},
                                 init = inits::from_vector(mask));
    return std::make_tuple(x, xMask);
  }

  template <class... Args>
  PoolingEncoder(Ptr<Config> options, Args... args)
    : EncoderBase(options, args...)
  {}

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch,
                          size_t encoderIndex) {
    auto wordEmbeddings = buildEmbeddings(graph, encoderIndex);
    auto posEmbeddings = buildPositionalEmbeddings(graph, encoderIndex);

    // select embeddings that occur in the batch
    Expr batchEmbeddings, batchMask;
    std::tie(batchEmbeddings, batchMask)
      = lookupWithPosition(wordEmbeddings, posEmbeddings, batch, encoderIndex);

    // apply dropout over source words
    float dropProb = inference_ ? 0 : opt<float>("dropout-src");
    if(dropProb) {
      int srcWords = batchEmbeddings->shape()[2];
      auto dropMask = graph->dropout(dropProb, {1, 1, srcWords});
      batchEmbeddings = dropout(batchEmbeddings, keywords::mask = dropMask);
    }

    Expr context = getContext(graph, batchEmbeddings, batchMask);

    return New<CNNEncoderState>(context, batchEmbeddings, batchMask, batch);
  }
};

class ConvNMT
    : public EncoderDecoder<PoolingEncoder, DecoderS2S> {
public:
  template <class... Args>
  ConvNMT(Ptr<Config> options, Args... args)
      : EncoderDecoder(options, args...)
  {}
};

}  // namespace marian
