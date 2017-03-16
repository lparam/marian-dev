
#include "marian.h"
#include "models/dl4mt.h"
#include "models/gnmt.h"
#include "models/multi_gnmt.h"
#include "models/conv_nmt.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv);;

  auto type = options->get<std::string>("type");
  if(type == "gnmt")
    Train<AsyncGraphGroup<GNMT>>(options);
  else if(type == "multi-gnmt")
    Train<AsyncGraphGroup<MultiGNMT>>(options);
  else if(type == "conv-nmt")
    Train<AsyncGraphGroup<ConvNMT>>(options);
  else
    Train<AsyncGraphGroup<DL4MT>>(options);

  return 0;
}
