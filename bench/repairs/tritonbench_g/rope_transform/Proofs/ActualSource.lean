import bench.repairs.tritonbench_g.rope_transform.Kernels
import bench.tritonbench_g.rope_transform.Proofs.TypedContract

/-! Actual copied source execution against the existing independent masked
memory formula. The historical typed source and its algorithm-erasure audit
remain separate; no second authoritative implementation is constructed here. -/
namespace VeriTile.Bench.Repairs.RopeTransform.ActualSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
open VeriTile.Bench.TritonBenchG.RopeTransform.Typed
noncomputable section
set_option maxHeartbeats 20000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
abbrev Parameters := VeriTile.Bench.TritonBenchG.RopeTransform.Typed.Parameters
abbrev source := Kernels.source

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
    simp [kernel, source, VeriTile.Bench.Repairs.RopeTransform.Kernels.source, FP.Structural.exec, ComputeKernel.surfaceBody,
      FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
      FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
      FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
      FP.Structural.numericLt, FP.Structural.bop, FP.Structural.natLt,
      FP.Structural.integralFloorDiv, FP.Structural.integralMod,
      FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind,
      FloatDType.numericDType, FloatDType.toTileDType, fold_writes_regs]
    try rfl

#axiomsClean execution

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    (execution M p s).val.mem = referenceMemory M .fp32 .fp32 p s := by
  rcases p with ⟨q,k,c,sn,qs,ks,cs,ss,len,nq,nk,hd,pq,pk,ph,back⟩
  cases back
  all_goals
    simp [execution, referenceMemory, scatter, laneResult, component,
      coefficient, paddedZero, readFloat, active, address, fold_writes_mem,
      ofFloat, toFloat, Region.cast, State.setReg,
      FloatDType.toTileDType, FloatDType.numericDType, Nat.add_assoc]

#axiomsClean execution_memory
end
end VeriTile.Bench.Repairs.RopeTransform.ActualSource
