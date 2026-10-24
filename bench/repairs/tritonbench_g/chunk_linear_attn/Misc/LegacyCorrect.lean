import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.Admission
import bench.repairs.tritonbench_g.chunk_linear_attn.Misc.ScalarBinding

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Correct
open VeriTile Triton FP.Structural
open Host
noncomputable section

/-- Complete forward tuple, six backward slots, saved context, source grids,
independent causal formula and VJP, all returned tensors, and physical frame.
The structural contract and pointer room are derived from actual host guards. -/
theorem wrapper_correct (c : Invocation) (flags : Admission.RuntimeFlags)
    (checked : Admission.check c flags=Bool.true) (api : Admission.TensorAPI c)
    (a : Allocations) (ac : AllocationContract c a)
    (p : Admission.PointerObservation c a) (pointers : Admission.checkPointers c a p=Bool.true)
    (separate : ∀ x∈outputBuffers a,∀ y∈outputBuffers a,x≠y → Physical.Separate x y)
    (fresh : ∀ x∈inputBuffers c a,∀ y∈outputBuffers a,Physical.Separate x y)
    (initial : InitialLayout c) (before : Nat → Cell ℝ) (fallback : Memory ℝ)
    (copies : CotangentCopies Real.algebra c a before) :
    RealPost c a before fallback (realPublished c a before fallback) :=
  real_complete c (Admission.contract c flags checked api) a ac
    (Admission.physicalAllocation c a p pointers separate fresh) initial before fallback copies

#axiomsClean wrapper_correct
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Correct
