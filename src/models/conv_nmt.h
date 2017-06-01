#pragma once

#include "models/encdec.h"
#include "layers/attention.h"
#include "layers/rnn.h"
#include "layers/convolution.h"

namespace marian {

typedef AttentionCell<GRU, GlobalAttention, GRU> CGRU;

class ConvolutionalEncoderState : public EncoderState {
  public:
    ConvolutionalEncoderState(Expr attContext, Expr srcContext, Expr mask,
                              Ptr<data::CorpusBatch> batch);

    Expr getSrcContext();
    Expr getContext();
    Expr getMask();

    const std::vector<size_t>& getSourceWords();

  private:
    Expr attContext_;
    Expr srcContext_;
    Expr mask_;
    Ptr<data::CorpusBatch> batch_;
};


class ConvolutionalEncoder : public EncoderBase {
  public:
    template <class ...Args>
    ConvolutionalEncoder(Ptr<Config> options, Args ...args)
     : EncoderBase(options, args...)
    {}

    Ptr<EncoderState>
    build(Ptr<ExpressionGraph> graph,
          Ptr<data::CorpusBatch> batch,
          size_t batchIdx = 0);

  protected:
    std::tuple<Expr, Expr>
    prepareSource(Expr emb,
                  Expr posEmb,
                  Ptr<data::CorpusBatch> batch,
                  size_t index);
};


class ConvolutionalDecoder : public DecoderBase {
  private:
    Ptr<GlobalAttention> attention_;
    Ptr<RNN<CGRU>> rnn;

  public:
    template <class ...Args>
    ConvolutionalDecoder(Ptr<Config> options, Args ...args)
     : DecoderBase(options, args...)
    {}

    Ptr<DecoderState> startState(Ptr<EncoderState> encState);

    virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                   Ptr<DecoderState> state);

};


class ConvNMT : public EncoderDecoder<ConvolutionalEncoder, ConvolutionalDecoder> {
  public:
    template <class ...Args>
    ConvNMT(Ptr<Config> options, Args ...args)
     : EncoderDecoder(options, args...) {
    }
};

}  // namespace marian
