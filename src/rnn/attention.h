#pragma once

#include "common/definitions.h"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "models/states.h"

#include "rnn/types.h"

namespace marian {

namespace rnn {

Expr attOps(Expr va, Expr context, Expr state, Expr coverage = nullptr);

class GlobalAttention : public CellInput {
private:
  Expr Wa_, ba_, Ua_, va_;

  Expr gammaContext_, betaContext_;
  Expr gammaState_, betaState_;

  Expr softmaxMask_;
  Expr mappedContext_;
  std::vector<Expr> contexts_;
  std::vector<Expr> alignments_;
  bool layerNorm_;
  float dropout_;

  Expr context_;
  Expr attended_;
  Expr mask_;

  Expr contextDropped_;
  Expr dropMaskContext_;
  Expr dropMaskState_;

public:

  GlobalAttention(Ptr<ExpressionGraph> graph,
                  Ptr<Options> options,
                  Expr context,
                  Expr attended,
                  Expr mask)
      : CellInput(options),
        context_(context),
        attended_(attended),
        mask_(mask),
        contextDropped_(context_) {

    int dimDecState = options_->get<int>("dimState");
    dropout_ = options_->get<float>("dropout");
    layerNorm_ = options_->get<bool>("layer-normalization");
    std::string prefix = options_->get<std::string>("prefix");

    int dimEncState = context_->shape()[1];

    Wa_ = graph->param(prefix + "_W_comb_att",
                       {dimDecState, dimEncState},
                       keywords::init = inits::glorot_uniform);
    Ua_ = graph->param(prefix + "_Wc_att",
                       {dimEncState, dimEncState},
                       keywords::init = inits::glorot_uniform);
    va_ = graph->param(prefix + "_U_att",
                       {dimEncState, 1},
                       keywords::init = inits::glorot_uniform);
    ba_ = graph->param(
        prefix + "_b_att", {1, dimEncState}, keywords::init = inits::zeros);

    if(dropout_ > 0.0f) {
      dropMaskContext_ = graph->dropout(dropout_, {1, dimEncState});
      dropMaskState_ = graph->dropout(dropout_, {1, dimDecState});
    }

    if(dropMaskContext_)
      contextDropped_
          = dropout(contextDropped_, keywords::mask = dropMaskContext_);

    if(layerNorm_) {
      gammaContext_ = graph->param(prefix + "_att_gamma1",
                                   {1, dimEncState},
                                   keywords::init = inits::from_value(1.0));
      gammaState_ = graph->param(prefix + "_att_gamma2",
                                 {1, dimEncState},
                                 keywords::init = inits::from_value(1.0));

      mappedContext_
          = layer_norm(dot(contextDropped_, Ua_), gammaContext_, ba_);
    } else {
      mappedContext_ = affine(contextDropped_, Ua_, ba_);
    }

    auto softmaxMask = mask_;
    if(softmaxMask) {
      Shape shape = {softmaxMask->shape()[2], softmaxMask->shape()[0]};
      softmaxMask_ = transpose(reshape(softmaxMask, shape));
    }
  }

  GlobalAttention(Ptr<ExpressionGraph> graph,
                  Ptr<Options> options,
                  Ptr<EncoderState> encState)
    : GlobalAttention(graph,
                      options,
                      encState->getContext(),
                      encState->getAttended(),
                      encState->getMask()) {}


  Expr apply(State state) {
    return apply(state.output);
  }

  Expr apply(Expr input) {
    using namespace keywords;

    int dimBatch = contextDropped_->shape()[0];
    int srcWords = contextDropped_->shape()[2];
    int dimBeam = input->shape()[3];


    if(dropMaskState_)
      input = dropout(input, keywords::mask = dropMaskState_);

    auto mappedState = dot(input, Wa_);
    if(layerNorm_)
      mappedState = layer_norm(mappedState, gammaState_);

    auto attReduce = attOps(va_, mappedContext_, mappedState);

    // @TODO: horrible ->
    auto e = reshape(transpose(softmax(transpose(attReduce), softmaxMask_)),
                     {dimBatch, 1, srcWords, dimBeam});
    // <- horrible

    auto alignedSource = weighted_average(attended_, e, axis = 2);

    contexts_.push_back(alignedSource);
    alignments_.push_back(e);
    return alignedSource;
  }

  std::vector<Expr>& getContexts() { return contexts_; }

  Expr getContext() {
    return concatenate(contexts_, keywords::axis=2);
  }

  std::vector<Expr>& getAlignments() { return alignments_; }

  virtual void clear() {
    contexts_.clear();
    alignments_.clear();
  }

  int dimOutput() { return context_->shape()[1]; }
};

using Attention = GlobalAttention;

}

}
