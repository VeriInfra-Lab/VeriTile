import bench.repairs.tritonbench_g.fused_recurrent_delta.Misc.HostCorrect
import bench.repairs.tritonbench_g.fused_recurrent_delta.Misc.HostFPEquiv

namespace VeriTile.Bench.Repairs.FusedRecurrentDelta
noncomputable section
/-- Ordinary completed wrapper call: independent mathematical result/VJP,
actual in-place v write, optional states, and complete logical frames. -/
theorem whole_correct (c : ABI.Call) (admitted : ABI.Admitted c) (gf gb : Trees)
    (trees : RealBackward.Legal c.shape gf gb) (initial : Memory ℝ) :
    ForwardClaim c.shape (ABI.flags c) (ABI.realScale c) (prepareBeta c.shape (ABI.flags c) Real.ops initial)
      (forwardCall c.shape (ABI.flags c) Real.ops gf (ABI.realScale c) initial) ∧
    BackwardClaim c.shape (ABI.flags c) (ABI.realScale c) (prepareBeta c.shape (ABI.flags c) Real.ops initial)
      (backwardCall c.shape (ABI.flags c) Real.ops gb (ABI.realScale c) (forwardCall c.shape (ABI.flags c) Real.ops gf (ABI.realScale c) initial)) ∧
    MemoryProof.Preserves (MemoryProof.ForwardCallWrites c.shape (ABI.flags c)) initial
      (forwardCall c.shape (ABI.flags c) Real.ops gf (ABI.realScale c) initial) ∧
    MemoryProof.Preserves (MemoryProof.BackwardWrites c.shape (ABI.flags c))
      (forwardCall c.shape (ABI.flags c) Real.ops gf (ABI.realScale c) initial)
      (backwardCall c.shape (ABI.flags c) Real.ops gb (ABI.realScale c) (forwardCall c.shape (ABI.flags c) Real.ops gf (ABI.realScale c) initial)) :=
  correct c.shape (ABI.flags c) gf gb trees (ABI.realScale c) initial
#axiomsClean whole_correct
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta
