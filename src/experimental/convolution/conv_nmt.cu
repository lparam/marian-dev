#include "models/amun.h"
#include "models/conv_nmt.h"
#include "models/encdec.h"

namespace marian {

ConvolutionalEncoderState::ConvolutionalEncoderState(
    Expr attContext, Expr srcContext, Expr mask, Ptr<data::CorpusBatch> batch)
    : attContext_(attContext),
      srcContext_(srcContext),
      mask_(mask),
      batch_(batch) {}

const std::vector<size_t>& ConvolutionalEncoderState::getSourceWords() {
  return batch_->front()->indeces();
}

Expr ConvolutionalEncoderState::getSrcContext() {
  return srcContext_;
}

Expr ConvolutionalEncoderState::getContext() {
  return attContext_;
}

Expr ConvolutionalEncoderState::getMask() {
  return mask_;
}

Ptr<EncoderState> ConvolutionalEncoder::build(Ptr<ExpressionGraph> graph,
                                              Ptr<data::CorpusBatch> batch,
                                              size_t batchIdx) {
  using namespace keywords;

  int dimSrcVoc = options_->get<std::vector<int>>("dim-vocabs")[batchIdx];
  int dimSrcEmb = options_->get<int>("dim-emb");
  int maxSrcLength = options_->get<int>("max-length");

  float dropoutSrc = inference_ ? 0 : options_->get<float>("dropout-src");

  auto xEmb = Embedding("Wemb", dimSrcVoc, dimSrcEmb)(graph);
  auto posEmb = Embedding("Wemb_pos", maxSrcLength, dimSrcEmb)(graph);

  Expr x, xMask;

  std::tie(x, xMask) = prepareSource(xEmb, posEmb, batch, batchIdx);

  std::string convType = options_->get<std::string>("conv-enc-type");

  if(dropoutSrc) {
    int srcWords = x->shape()[2];
    auto srcWordDrop = graph->dropout(dropoutSrc, {1, 1, srcWords});
    x = dropout(x, mask = srcWordDrop);
  }

  Expr attContext;
  Expr srcContext;
  if(convType == "pooling") {
    attContext = Pooling("enc_pooling")(x, xMask);
    srcContext = x;
  } else if(convType == "full") {
    attContext = Convolution("conv_att", 3, 1, 1)(x, xMask);
    srcContext = Convolution("conv_src", 3, 1, 1)(x, xMask);
  } else {
    LOG("Unknown type of convolutional encoder");
  }

  return New<ConvolutionalEncoderState>(attContext, srcContext, xMask, batch);
}

std::tuple<Expr, Expr> ConvolutionalEncoder::prepareSource(
    Expr emb, Expr posEmb, Ptr<data::CorpusBatch> batch, size_t index) {
  using namespace keywords;

  auto& wordIndeces = batch->at(index)->indeces();

  auto& mask = batch->at(index)->mask();

  std::vector<size_t> posIndeces;

  for(size_t iPos = 0; iPos < batch->at(index)->batchWidth(); ++iPos) {
    for(size_t i = 0; i < batch->at(index)->batchSize(); ++i) {
      if(iPos < posEmb->shape()[0]) {
        posIndeces.push_back(iPos);
      } else {
        posIndeces.push_back(posEmb->shape()[0] - 1);
      }
    }
  }

  int batchSize = batch->size();
  int dimEmb = emb->shape()[1];
  int batchLength = batch->at(index)->batchWidth();

  auto graph = emb->graph();

  auto xWord
      = reshape(rows(emb, wordIndeces), {batchSize, dimEmb, batchLength});
  auto xPos
      = reshape(rows(posEmb, posIndeces), {batchSize, dimEmb, batchLength});
  auto x = xWord + xPos;
  auto xMask = graph->constant(shape = {batchSize, 1, batchLength},
                               init = inits::from_vector(mask));
  return std::make_tuple(x, xMask);
}

Ptr<DecoderState> ConvolutionalDecoder::startState(Ptr<EncoderState> encState) {
  using namespace keywords;

  auto meanContext = weighted_average(
      std::static_pointer_cast<ConvolutionalEncoderState>(encState)
          ->getSrcContext(),
      encState->getMask(),
      axis = 2);

  bool layerNorm = options_->get<bool>("layer-normalization");
  auto start = Dense("ff_state",
                     options_->get<int>("dim-rnn"),
                     activation = act::tanh,
                     normalize = layerNorm)(meanContext);
  std::vector<Expr> startStates(options_->get<size_t>("layers-dec"), start);
  return New<DecoderStateAmun>(start, nullptr, encState);
}

Ptr<DecoderState> ConvolutionalDecoder::step(Ptr<ExpressionGraph> graph,
                                             Ptr<DecoderState> state) {
  using namespace keywords;

  int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").back();
  int dimTrgEmb = options_->get<int>("dim-emb");
  int dimDecState = options_->get<int>("dim-rnn");
  bool layerNorm = options_->get<bool>("layer-normalization");
  bool skipDepth = options_->get<bool>("skip");
  size_t decoderLayers = options_->get<size_t>("layers-dec");

  float dropoutRnn = inference_ ? 0 : options_->get<float>("dropout-rnn");
  float dropoutTrg = inference_ ? 0 : options_->get<float>("dropout-trg");

  auto stateAmun = std::dynamic_pointer_cast<DecoderStateAmun>(state);
  auto embeddings = stateAmun->getTargetEmbeddings();

  if(dropoutTrg) {
    int trgWords = embeddings->shape()[2];
    auto trgWordDrop = graph->dropout(dropoutTrg, {1, 1, trgWords});
    embeddings = dropout(embeddings, mask = trgWordDrop);
  }

  if(!attention_) {
    attention_ = New<GlobalAttention>(
        "decoder",
        state->getEncoderState(),
        std::static_pointer_cast<ConvolutionalEncoderState>(
            state->getEncoderState())
            ->getSrcContext(),
        dimDecState,
        dropout_prob = dropoutRnn,
        normalize = layerNorm);
  }

  if(!rnn)
    rnn = New<RNN<CGRU>>(graph,
                         "decoder",
                         dimTrgEmb,
                         dimDecState,
                         attention_,
                         dropout_prob = dropoutRnn,
                         normalize = layerNorm);
  auto stateOut = (*rnn)(embeddings, stateAmun->getState());

  bool single = stateAmun->doSingleStep();

  auto alignedContextsVec = attention_->getContexts();
  auto alignedContext = single ?
                            alignedContextsVec.back() :
                            concatenate(alignedContextsVec, keywords::axis = 2);

  //// 2-layer feedforward network for outputs and cost
  auto logitsL1 = Dense(
      "ff_logit_l1", dimTrgEmb, activation = act::tanh, normalize = layerNorm)(
      embeddings, stateOut, alignedContext);

  auto logitsOut = Dense("ff_logit_l2", dimTrgVoc)(logitsL1);

  return New<DecoderStateAmun>(stateOut, logitsOut, state->getEncoderState());
}

}  // namespace marian
