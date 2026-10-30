import bench.tritonbench_g.rope_transform.Proofs.Mathematical_TypedSource
import bench.tritonbench_g.isfinite_kernel.Proofs.Mathematical_MemoryProjection
import VeriTile.Triton.Float.ScalarBackend

/-! Complete typed rotary execution against an independent operation-order
reference. Multiplications, additions/subtractions and both storage conversions
are distinct backend calls. No real-number identity or final-output assumption.
The backend represents the unfused source graph; a GPU compiler's fusion and
format policies still need an explicit binding. -/
namespace VeriTile.Bench.TritonBenchG.RopeTransform.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
noncomputable section
set_option maxHeartbeats 20000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

structure Parameters where
  (q k cos sin : RegionName)
  (qStride kStride cosStride sinStride length
    qHeads kHeads headDim paddedQHeads paddedKHeads paddedHeadDim : Nat)
  backward : Bool

def kernel (input rotation : FloatDType) (p : Parameters) : ComputeKernel :=
  source input rotation p.q p.k p.cos p.sin
    p.qStride p.kStride p.cosStride p.sinStride p.length 0
    p.qHeads p.kHeads p.headDim p.paddedQHeads p.paddedKHeads p.paddedHeadDim 0 p.backward

/-- Execution includes both branches, all four stores, tails and arbitrary
initial register contents. This is totality, not by itself the value contract. -/
def execution {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (kernel .fp32 .fp32 p) s = some t} := by
  rcases p with ⟨q,k,c,sn,qs,ks,cs,ss,len,nq,nk,hd,pq,pk,ph,back⟩
  cases back
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp [kernel, source, FP.Structural.exec, ComputeKernel.surfaceBody,
      FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
      FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
      FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
      FP.Structural.numericLt, FP.Structural.bop, FP.Structural.natLt,
      FP.Structural.integralFloorDiv, FP.Structural.integralMod,
      FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind,
      FloatDType.numericDType, FloatDType.toTileDType, fold_writes_regs]
    try rfl

#axiomsClean execution
end
end VeriTile.Bench.TritonBenchG.RopeTransform.Typed
