#pragma once

#include "common/definitions.h"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "models/states.h"

#include "rnn/types.h"

namespace marian {

namespace rnn {

Expr attOps(Expr va, Expr context, Expr state);

class GlobalAttention : public CellInput {
private:
  Expr Wa_, ba_, Ua_, va_;

  Expr gammaKeys_;
  Expr gammaQuery_;

  Expr softmaxMask_;
  Expr mappedKeys_;
  std::vector<Expr> contexts_;
  std::vector<Expr> alignments_;
  bool layerNorm_;
  float dropout_;

  Expr keys_;
  Expr values_;
  Expr mask_;

  Expr keysDropped_;
  Expr dropMaskKeys_;
  Expr dropMaskQuery_;

public:

  GlobalAttention(Ptr<ExpressionGraph> graph,
                  Ptr<Options> options,
                  Expr keys,
                  Expr values,
                  Expr mask)
      : CellInput(options),
        keys_(keys),
        values_(values),
        mask_(mask),
        keysDropped_(keys_) {

    // @TODO: replace dimState with dimQuery everywhere
    int dimQuery = options_->get<int>("dimState");
    dropout_ = options_->get<float>("dropout");
    layerNorm_ = options_->get<bool>("layer-normalization");
    std::string prefix = options_->get<std::string>("prefix");

    int dimKeys = keys_->shape()[1];

    Wa_ = graph->param(prefix + "_W_comb_att",
                       {dimQuery, dimKeys},
                       keywords::init = inits::glorot_uniform);
    Ua_ = graph->param(prefix + "_Wc_att",
                       {dimKeys, dimKeys},
                       keywords::init = inits::glorot_uniform);
    va_ = graph->param(prefix + "_U_att",
                       {dimKeys, 1},
                       keywords::init = inits::glorot_uniform);
    ba_ = graph->param(
        prefix + "_b_att", {1, dimKeys}, keywords::init = inits::zeros);

    if(dropout_ > 0.0f) {
      dropMaskKeys_ = graph->dropout(dropout_, {1, dimKeys});
      dropMaskQuery_ = graph->dropout(dropout_, {1, dimQuery});
    }

    if(dropMaskKeys_)
      keysDropped_ = dropout(keysDropped_, keywords::mask = dropMaskKeys_);

    if(layerNorm_) {
      gammaKeys_ = graph->param(prefix + "_att_gamma1",
                                {1, dimKeys},
                                keywords::init = inits::from_value(1.0));
      gammaQuery_ = graph->param(prefix + "_att_gamma2",
                                 {1, dimKeys},
                                 keywords::init = inits::from_value(1.0));

      mappedKeys_
          = layer_norm(dot(keysDropped_, Ua_), gammaKeys_, ba_);
    } else {
      mappedKeys_ = affine(keysDropped_, Ua_, ba_);
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

  Expr apply(Expr query) {
    using namespace keywords;

    int dimBatch = keysDropped_->shape()[0];
    int dimTimeKeys = keysDropped_->shape()[2];
    int dimTimeQuery = query->shape()[2];
    int dimBeam = query->shape()[3];

    //debug(query, "query");

    if(dropMaskQuery_)
      query = dropout(query, keywords::mask = dropMaskQuery_);

    auto mappedQuery = dot(query, Wa_);
    if(layerNorm_)
      mappedQuery = layer_norm(mappedQuery, gammaQuery_);

    auto scores = attOps(va_, mappedKeys_, mappedQuery);
    //debug(scores, "scores");

    // @TODO: horrible ->
    auto weights = reshape(transpose(softmax(transpose(scores), softmaxMask_)),
                           {dimBatch, 1, dimTimeKeys, dimTimeQuery * dimBeam});
    //debug(weights, "weights");
    // <- horrible

    auto output = weighted_average(values_, weights, axis = 2);
    output = reshape(output, {dimBatch, output->shape()[1], dimTimeQuery, 1});
    //std::cerr << output->shape() << std::endl;
    //debug(output, "outputs");

    contexts_.push_back(output);
    alignments_.push_back(weights);
    return output;
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

  int dimOutput() { return keys_->shape()[1]; }
};

using Attention = GlobalAttention;

}

}
