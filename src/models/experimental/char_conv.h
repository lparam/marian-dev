#pragma once

#include "models/encdec.h"
#include "models/s2s.h"
#include "layers/convolution.h"
#include "layers/highway.h"

namespace marian {

class CharConvEncoder : public EncoderBase {
public:
  template <class... Args>
  CharConvEncoder(Ptr<Config> options, Args... args)
      : EncoderBase(options, args...) {}

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch,
                          size_t batchIdx) {
    using namespace keywords;

    int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIdx];
    int dimEmb = opt<std::vector<int>>("dim-emb")[batchIdx];

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

    auto embeddings = embFactory.construct();

    Expr batchEmbeddings, batchMask;
    std::tie(batchEmbeddings, batchMask)
      = EncoderBase::lookup(embeddings, batch, batchIdx);

    float dropProb = inference_ ? 0 : opt<float>("dropout-src");
    if(dropProb) {
      int srcWords = batchEmbeddings->shape()[2];
      auto dropMask = graph->dropout(dropProb, {1, 1, srcWords});
      batchEmbeddings = dropout(batchEmbeddings, mask = dropMask);
    }

    std::vector<int> convWidths({1, 2, 3, 4, 5, 6, 7, 8});
    std::vector<int> convSizes({200, 200, 200, 200, 200, 200, 200, 200});
    int stride = 5;

    auto convolution = MultiConvolution("multi_conv", dimEmb, convWidths, convSizes)
      (batchEmbeddings, batchMask);
    auto highway = Highway("highway", 4)(convolution);
    Expr context = applyEncoderRNN(graph, batchEmbeddings, batchMask, batchIdx);
    Expr stridedMask = getStridedMask(graph, batch, batchIdx, stride);

    return New<EncoderState>(context, stridedMask, batch);
  }

protected:
  Expr applyEncoderRNN(Ptr<ExpressionGraph> graph,
                       Expr embeddings, Expr mask,
                       size_t encIdx) {
    using namespace keywords;
    float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");

    size_t embDim = embeddings->shape()[1];

    auto rnnFw = rnn::rnn(graph)
                 ("type", opt<std::string>("enc-cell"))
                 ("direction", rnn::dir::forward)
                 ("dimInput", embDim)
                 ("dimState", opt<std::vector<int>>("dim-rnn")[encIdx])
                 ("dropout", dropoutRnn)
                 ("layer-normalization", opt<bool>("layer-normalization"))
                 ("skip", opt<bool>("skip"));

    auto stacked = rnn::stacked_cell(graph);
    stacked.push_back(rnn::cell(graph)("prefix", prefix_ + "_bi"));
    rnnFw.push_back(stacked);

    auto rnnBw = rnn::rnn(graph)
                 ("type", opt<std::string>("enc-cell"))
                 ("direction", rnn::dir::backward)
                 ("dimInput", embDim)
                 ("dimState", opt<std::vector<int>>("dim-rnn")[encIdx])
                 ("dropout", dropoutRnn)
                 ("layer-normalization", opt<bool>("layer-normalization"))
                 ("skip", opt<bool>("skip"));

    auto stackedBack = rnn::stacked_cell(graph);
    stackedBack.push_back(rnn::cell(graph)("prefix", prefix_ + "_bi_r"));
    rnnBw.push_back(stackedBack);

    auto context = concatenate({rnnFw->transduce(embeddings, mask),
                                rnnBw->transduce(embeddings, mask)},
                                axis=1);
    return context;
  }

  Expr getStridedMask(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch,
      size_t index, int stride) {
    auto subBatch = (*batch)[index];

    int dimBatch = subBatch->batchSize();

    std::vector<float> strided;
    for (size_t wordIdx = 0; wordIdx < subBatch->mask().size(); wordIdx += stride * dimBatch) {
      for (size_t j = wordIdx; j < wordIdx + dimBatch; ++j) {
        strided.push_back(subBatch->mask()[j]);
      }
    }
    int dimWords = strided.size() / dimBatch;
    auto stridedMask = graph->constant({dimBatch, 1, dimWords},
                                       keywords::init = inits::from_vector(strided));
    return stridedMask;
  }

};

typedef EncoderDecoder<CharConvEncoder, DecoderS2S> CharConvModel;


}
