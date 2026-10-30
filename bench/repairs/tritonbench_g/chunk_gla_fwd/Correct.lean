import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.LogicalIO

/-! Public finite IO for the two separate calls. RealIO exposes the five
shared sources, complete grids, initialization, channels and private scratch.
Operational helpers live in Proofs/; historical raw-memory statements live in Misc/. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.Correct
open VeriTile Triton
open scoped VeriTile.Triton.LogicalIO
open scoped BigOperators
noncomputable section

/- Independent scalar mathematics and the complete logical output channels.
These definitions are definitionally connected to the canonical target used below. -/
namespace Mathematics
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
def attention (c : Host.Dimensions) (a : RealIO.LogicalIO.ProducerInputs c) (scale : ℝ)
    (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) : ℝ :=
  if j.val ≤ t.val % c.BT then
    (∑ k ∈ Finset.range c.K,
      RealIO.LogicalIO.producerQuery c a h.val t.val k *
        RealIO.LogicalIO.producerKey c a h.val (t.val/c.BT*c.BT+j.val) k *
        Real.exp (RealIO.LogicalIO.producerGate c a h.val t.val k -
          RealIO.LogicalIO.producerGate c a h.val (t.val/c.BT*c.BT+j.val) k)) * scale
  else 0

def consumerValue (c : Host.Dimensions) (a : RealIO.LogicalIO.ConsumerInputs c) (scale : ℝ)
    (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) : ℝ :=
  (∑ k ∈ Finset.range c.K,
    RealIO.LogicalIO.consumerQuery c a h.val t.val k * scale *
      Real.exp (RealIO.LogicalIO.consumerGate c a h.val t.val k) *
      RealIO.LogicalIO.consumerHistory c a h.val (t.val/c.BT) k v.val) +
  ∑ j ∈ Finset.range (t.val%c.BT+1),
    RealIO.LogicalIO.consumerAttention c a h.val t.val j *
      RealIO.LogicalIO.consumerValue c a h.val (t.val/c.BT*c.BT+j) v.val

def producerResult (c : Host.Dimensions) (scale : ℝ) := fun values => attention c values scale

def consumerResult (c : Host.Dimensions) (scale : ℝ) := fun values => consumerValue c values scale

end Mathematics


/-- Full producer, including both K branches and every rectangular attention
cell, implements the independent causal attention formula. -/
specification producer_kernel_correctness (c : Host.Dimensions) (scale : ℝ) :
    Spec.Real (RealIO.LogicalIO.producer c scale ⊨ Mathematics.producerResult c scale) :=
  RealIO.LogicalIO.producer_correct c scale

/-- Full standalone consumer implements the independent gated history sum
and causal attention/value contraction over caller-supplied attention/history. -/
specification consumer_kernel_correctness (c : Host.Dimensions) (scale : ℝ) :
    Spec.Real (RealIO.LogicalIO.consumer c scale ⊨ Mathematics.consumerResult c scale) :=
  RealIO.LogicalIO.consumer_correct c scale

#axiomsClean producer_kernel_correctness
#axiomsClean consumer_kernel_correctness
#stmtSurfaceSubset producer_kernel_correctness ⊆ [Spec.Real,Host.Dimensions,
  VeriTile.Triton.LogicalIO.Implements,AlgorithmFillProgramIO.Implements,RealIO.producer,
  RealIO.ProducerValues,Host.attentionSize,RealIO.LogicalIO.ProducerInputs,RealIO.LogicalIO.producer,
  Host.Dimensions.heads,Host.Dimensions.T,Host.Dimensions.BT,Mathematics.producerResult]
#stmtSurfaceSubset consumer_kernel_correctness ⊆ [Spec.Real,Host.Dimensions,
  VeriTile.Triton.LogicalIO.Implements,AlgorithmFillProgramIO.Implements,RealIO.consumer,
  RealIO.ConsumerValues,Host.outputSize,RealIO.LogicalIO.ConsumerInputs,RealIO.LogicalIO.consumer,
  Host.Dimensions.heads,Host.Dimensions.T,Host.Dimensions.V,Mathematics.consumerResult]
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.Correct
