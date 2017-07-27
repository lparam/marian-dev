#include "rnn/attention.h"

#include "graph/node_operators_binary.h"
#include "kernels/tensor_operators.h"

namespace marian {

namespace rnn {

struct AttentionNodeOp : public NaryNodeOp {
  AttentionNodeOp(const std::vector<Expr>& nodes)
      : NaryNodeOp(nodes, keywords::shape = newShape(nodes)) {}

  Shape newShape(const std::vector<Expr>& nodes) {
    Shape shape = nodes[1]->shape();

    Shape vaShape = nodes[0]->shape();
    Shape keysShape = nodes[1]->shape();
    Shape queryShape = nodes[2]->shape();

    for(int i = 0; i < 2; ++i) {
      UTIL_THROW_IF2(keysShape[i] != queryShape[i]
                     && keysShape[i] != 1
                     && queryShape[i] != 1,
                     "Shapes cannot be broadcasted");
      shape.set(i, std::max(keysShape[i], queryShape[i]));
    }

    UTIL_THROW_IF2(vaShape[0] != shape[1] || vaShape[1] != 1, "Wrong size");

    shape.set(1, 1);
    shape.set(2, keysShape[2]);
    shape.set(3, queryShape[2] * queryShape[3]);

    return shape;
  }

  NodeOps forwardOps() {
    return {NodeOp(Att(val_,
                       child(0)->val(),
                       child(1)->val(),
                       child(2)->val()))};
  }

  NodeOps backwardOps() {
    return {
      NodeOp(AttBack(child(0)->grad(),
                     child(1)->grad(),
                     child(2)->grad(),
                     child(0)->val(),
                     child(1)->val(),
                     child(2)->val(),
                     adj_);)
    };
  }

  // do not check if node is trainable
  virtual void runBackward(const NodeOps& ops) {
    for(auto&& op : ops)
      op();
  }

  const std::string type() { return "Att-ops"; }

  const std::string color() { return "yellow"; }
};

Expr attOps(Expr va, Expr keys, Expr query) {
  std::vector<Expr> nodes{va, keys, query};

  int dimBatch = keys->shape()[0];
  int dimWords = keys->shape()[2];
  int dimBeam = query->shape()[2] * query->shape()[3];
  return reshape(Expression<AttentionNodeOp>(nodes),
                 {dimWords, dimBatch, 1, dimBeam});
}

}
}
