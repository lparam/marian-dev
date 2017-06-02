#include <iostream>
#include <math.h>

#include "marian.h"
#include "layers/highway.h"

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

int main(int argc, char** argv) {
  auto config = Config(argc, argv, false, true);
  auto graph = New<ExpressionGraph>(false);
  graph->setDevice(0);
  graph->reserveWorkspaceMB(128);

  int dimBatch = 2;
  int dimWord = 4;
  int batchLength = 5;
  int numLayers = 1;

  int elemNum = dimBatch * dimWord * batchLength * numLayers;

  std::vector<float> embData(elemNum);
  std::vector<float> embMask(elemNum);

  for (size_t i = 0; i < embData.size(); ++i) {
    embData[i] =  1 / (float(i) + 1);
    if (i < dimBatch * batchLength) {
      embMask[i] = 1;
    }
  }

  auto x = graph->param("x", {dimBatch, dimWord, batchLength},
                        keywords::init=inits::from_vector(embData));

  auto output = Highway("highway", 4)(x);

  debug(x, "X");
  debug(output, "wghiway");

  graph->forward();
  graph->backward();
  return 0;
}
