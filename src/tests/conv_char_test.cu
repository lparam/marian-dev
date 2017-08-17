#include <iostream>
#include <math.h>

#include "marian.h"
#include "layers/highway.h"
#include "layers/convolution.h"

using namespace marian;

bool test_vectors(const std::vector<float>& output, const std::vector<float>& corrent) {
  if (output.size() != corrent.size()) {
    return false;
  }

  for (size_t i = 0; i < output.size(); ++i) {
    if (fabsf(output[i] - corrent[i]) > 0.0001f) {
      return false;
    }
  }
  return true;
}

std::vector<float> strideMask(const std::vector<float>& mask, int batchSize, int stride) {
  std::vector<float> strided;

  for (size_t wordIdx = 0; wordIdx < mask.size(); wordIdx += stride * batchSize) {
    for (size_t j = wordIdx; j < wordIdx + batchSize; ++j) {
      strided.push_back(mask[j]);
    }
  }
  return strided;
}

int main(int argc, char** argv) {
  auto graph = New<ExpressionGraph>();
  graph->setDevice(1);
  graph->reserveWorkspaceMB(128);

  int dimBatch = 2;
  int dimWord = 2;
  int batchLength = 7;
  int numLayers = 1;

  int elemNum = dimBatch * dimWord * batchLength * numLayers;

  std::vector<float> embData(elemNum);
  std::vector<float> embMask(dimBatch * batchLength);

  for (size_t i = 0; i < embData.size() ; ++i) {
    // embData[2 * dimWord * (i / dimWord) + (i % dimWord)] = float(1);// / (i + 1.0f);
    // embData[2 * dimWord * (i / dimWord) + (i % dimWord) + dimWord] = float(1); // / (i + 1.0f);
    embData[i] = float(i); // / (i + 1.0f);
  }

  for (auto& v : embMask) {
    v = 1.0f;
  }
  embMask.back() = 0.0f;

  auto x = graph->param("x", {dimBatch, dimWord, batchLength},
                        keywords::init=inits::from_vector(embData));


  auto xMask = graph->constant({dimBatch, 1, batchLength},
                               keywords::init=inits::from_vector(embMask));

  std::vector<int> convWidths({1, 2});
  std::vector<int> convSizes({1, 1});

  auto convolution = MultiConvolution("multi_conv", dimWord, convWidths, convSizes)(x, xMask);

  auto highway = Highway("highway", 4)(convolution);
  auto idx = graph->constant({120, 1}, keywords::init=inits::zeros);
  auto ce = cross_entropy(highway, idx);
  auto cost = mean(sum(ce, keywords::axis=2), keywords::axis=0);


  debug(x, "x");
  debug(cost, "COST");
  debug(convolution, "CONVOLUTION");
  debug(highway, "highway");
  debug(ce, "ce");

  graph->forward();
  graph->backward();

  std::vector<float> output;
  std::vector<float> output2;

  for (auto v : output2) std::cerr << v << " ";
  std::cerr << std::endl;

  return 0;
}
