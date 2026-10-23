import bench.repairs.tritonbench_g.chunk_retention_ops.Contract
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Contract

structure KV where
  dims : Dimensions
  k : Tensor
  v : Tensor

def KVContract (a : KV) : Prop := ShapeContract a.dims ∧
  TensorContract a.k (keyShape a.dims) a.k.metadata.device Bool.true Bool.true ∧
  TensorContract a.v (valueShape a.dims) a.k.metadata.device Bool.true Bool.true
noncomputable def kvCheck (a : KV) : Bool := shapeCheck a.dims &&
  tensorCheck a.k (keyShape a.dims) a.k.metadata.device Bool.true Bool.true &&
  tensorCheck a.v (valueShape a.dims) a.k.metadata.device Bool.true Bool.true

structure Primal extends KV where
  q : Tensor

def PrimalContract (a : Primal) : Prop := KVContract a.toKV ∧
  TensorContract a.q (keyShape a.dims) a.k.metadata.device Bool.true Bool.true
noncomputable def primalCheck (a : Primal) : Bool := kvCheck a.toKV &&
  tensorCheck a.q (keyShape a.dims) a.k.metadata.device Bool.true Bool.true

structure BT where
  exactInt : Bool
  value : Int
def BTContract (a : BT) : Prop := a.exactInt=Bool.true ∧ a.value=64
def btCheck (a : BT) : Bool := a.exactInt && decide (a.value=64)

structure ForwardH extends KV where
  bt : BT
  initial : Option Tensor
  finalIsBool : Bool
  final : Bool
def ForwardHContract (a : ForwardH) : Prop := KVContract a.toKV ∧ BTContract a.bt ∧
  (∀ t∈a.initial.toList,TensorContract t (stateShape a.dims) a.k.metadata.device Bool.true Bool.true) ∧
  a.finalIsBool=Bool.true
noncomputable def forwardHCheck (a : ForwardH) : Bool := kvCheck a.toKV && btCheck a.bt &&
  (a.initial.toList.all fun t => tensorCheck t (stateShape a.dims) a.k.metadata.device Bool.true Bool.true) &&
  a.finalIsBool

structure ForwardO extends Primal where
  bt : BT
  scale : Scalar
  history : Tensor
def ForwardOContract (a : ForwardO) : Prop := PrimalContract a.toPrimal ∧ BTContract a.bt ∧
  ScaleContract a.scale ∧ TensorContract a.history (historyShape a.dims) a.k.metadata.device Bool.true Bool.true
noncomputable def forwardOCheck (a : ForwardO) : Bool := primalCheck a.toPrimal && btCheck a.bt &&
  scaleCheck a.scale && tensorCheck a.history (historyShape a.dims) a.k.metadata.device Bool.true Bool.true

structure BackwardH extends Primal where
  bt : BT
  scale : Scalar
  incoming : Tensor
  terminal : Option Tensor
  initialGradient : Option Tensor
def backwardHReads (a : BackwardH) := [a.q,a.k,a.v,a.incoming]++a.terminal.toList
def BackwardHContract (a : BackwardH) : Prop := PrimalContract a.toPrimal ∧ BTContract a.bt ∧
  ScaleContract a.scale ∧ TensorContract a.incoming (valueShape a.dims) a.k.metadata.device Bool.false Bool.true ∧
  (∀ t∈a.terminal.toList,TensorContract t (stateShape a.dims) a.k.metadata.device Bool.false Bool.true) ∧
  (∀ t∈a.initialGradient.toList,TensorContract t (stateShape a.dims) a.k.metadata.device Bool.true Bool.true ∧
    ∀ input∈backwardHReads a,disjoint t input)
noncomputable def backwardHCheck (a : BackwardH) : Bool :=
  primalCheck a.toPrimal && btCheck a.bt && scaleCheck a.scale &&
  tensorCheck a.incoming (valueShape a.dims) a.k.metadata.device Bool.false Bool.true &&
  (a.terminal.toList.all fun t => tensorCheck t (stateShape a.dims) a.k.metadata.device Bool.false Bool.true) &&
  (a.initialGradient.toList.all fun t => tensorCheck t (stateShape a.dims) a.k.metadata.device Bool.true Bool.true &&
    (backwardHReads a).all fun input => !overlapCheck t input)

structure BackwardQKV extends Primal where
  scale : Scalar
  incoming : Tensor
  history : Tensor
  dhistory : Tensor
def BackwardQKVContract (a : BackwardQKV) : Prop := PrimalContract a.toPrimal ∧ ScaleContract a.scale ∧
  TensorContract a.incoming (valueShape a.dims) a.k.metadata.device Bool.false Bool.true ∧
  TensorContract a.history (historyShape a.dims) a.k.metadata.device Bool.true Bool.true ∧
  TensorContract a.dhistory (historyShape a.dims) a.k.metadata.device Bool.true Bool.true
noncomputable def backwardQKVCheck (a : BackwardQKV) : Bool := primalCheck a.toPrimal && scaleCheck a.scale &&
  tensorCheck a.incoming (valueShape a.dims) a.k.metadata.device Bool.false Bool.true &&
  tensorCheck a.history (historyShape a.dims) a.k.metadata.device Bool.true Bool.true &&
  tensorCheck a.dhistory (historyShape a.dims) a.k.metadata.device Bool.true Bool.true
end VeriTile.Bench.Repairs.ChunkRetentionOps.Contract
