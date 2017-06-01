#include "layers/rnn.h"


namespace marian {

Expr gruOps(const std::vector<Expr>& nodes, bool final) {
  return Expression<GRUFastNodeOp>(nodes, final);
}

}
