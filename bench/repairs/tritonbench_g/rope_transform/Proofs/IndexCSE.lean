import bench.repairs.tritonbench_g.rope_transform.Kernels
import bench.repairs.tritonbench_g.rope_transform.Proofs.Index
import bench.tritonbench_g.isfinite_kernel.Proofs.MemoryProjection
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.RopeTransform.IndexCSE
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
noncomputable section
set_option maxHeartbeats 20000000
set_option maxRecDepth 30000
set_option maxSynthPendingDepth 3000
set_option linter.unusedSimpArgs false

def original (p : T.Parameters) : ComputeKernel :=
  Kernels.source .fp32 .fp32 p.q p.k p.cos p.sin
    p.qStride p.kStride p.cosStride p.sinStride p.length 0
    p.qHeads p.kHeads p.headDim p.paddedQHeads p.paddedKHeads p.paddedHeadDim 0 p.backward

def candidate (p : T.Parameters) : ComputeKernel :=
  Kernels.indexShared .fp32 .fp32 p.q p.k p.cos p.sin
    p.qStride p.kStride p.cosStride p.sinStride p.length 0
    p.qHeads p.kHeads p.headDim p.paddedQHeads p.paddedKHeads p.paddedHeadDim 0 p.backward

theorem original_binding (p : T.Parameters) :
    original p = T.kernel .fp32 .fp32 p := rfl

/-- Proof-only split before the unchanged floating graph and stores. -/
def sourcePrefix (p : T.Parameters) := (original p).surfaceBody.take 14
def candidatePrefix (p : T.Parameters) := (candidate p).surfaceBody.take 14
def suffix (p : T.Parameters) := (original p).surfaceBody.drop 14

theorem candidate_body (p : T.Parameters) :
    (candidate p).surfaceBody = candidatePrefix p ++ suffix p := by
  cases p.backward <;> rfl

theorem prefix_execution {α : Type} [Inhabited α] (M : Algebra α)
    (p : T.Parameters) (s : State α) :
    run M (sourcePrefix p) s = run M (candidatePrefix p) s := by
  rcases p with ⟨q,k,c,sn,qs,ks,cs,ss,len,nq,nk,hd,pq,pk,ph,back⟩
  cases back
  all_goals
    simp [sourcePrefix,candidatePrefix,original,candidate,Kernels.source,Kernels.indexShared,
      ComputeKernel.surfaceBody,List.take,
      FP.Structural.run,FP.Structural.step,FP.Structural.evalExpr,
      FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.integralFloorDiv,FP.Structural.integralMod,
      FP.Structural.ofFloat,FP.Structural.toFloat,Region.cast,Option.bind,
      FloatDType.numericDType,FloatDType.toTileDType,fold_writes_regs]
    try rfl

theorem prefix_safe {α : Type} [Inhabited α] (M : Algebra α)
    (p : T.Parameters) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (sourcePrefix p) s ↔
      Safety.traceSafeList M bounds (candidatePrefix p) s := by
  rcases p with ⟨q,k,c,sn,qs,ks,cs,ss,len,nq,nk,hd,pq,pk,ph,back⟩
  cases back
  all_goals
    simp (config := { maxSteps := 200000 }) [sourcePrefix,candidatePrefix,
      original,candidate,Kernels.source,Kernels.indexShared,ComputeKernel.surfaceBody,List.take,
      Safety.traceSafeList_cons,Safety.traceSafeList_nil,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      FP.Structural.run,FP.Structural.step,FP.Structural.evalExpr,
      FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.integralFloorDiv,FP.Structural.integralMod,
      FP.Structural.ofFloat,FP.Structural.toFloat,Region.cast,Option.bind,
      FloatDType.numericDType,FloatDType.toTileDType,fold_writes_regs]
    try rfl

/-- Exact full-state equality for every scalar backend and all floating payloads. -/
theorem execution {α : Type} [Inhabited α] (M : Algebra α)
    (p : T.Parameters) (s : State α) :
    FP.Structural.exec M (original p) s = FP.Structural.exec M (candidate p) s := by
  change run M (original p).surfaceBody s = run M (candidate p).surfaceBody s
  rw [candidate_body]
  conv_lhs => rw [← List.take_append_drop 14 (original p).surfaceBody]
  change run M (sourcePrefix p ++ suffix p) s = _
  rw [run_append,run_append,prefix_execution]

/-- The exact same active memory accesses are safe before and after index CSE. -/
theorem trace_safe {α : Type} [Inhabited α] (M : Algebra α)
    (p : T.Parameters) (bounds : RegionName → Nat) (s : State α) :
    TraceSafe M bounds (original p) s ↔ TraceSafe M bounds (candidate p) s := by
  unfold TraceSafe
  rw [candidate_body]
  conv_lhs => rw [← List.take_append_drop 14 (original p).surfaceBody]
  change Safety.traceSafeList M bounds (sourcePrefix p ++ suffix p) s ↔ _
  simp only [Safety.traceSafeList_append,prefix_execution,prefix_safe]

#axiomsClean execution
#axiomsClean trace_safe
end
end VeriTile.Bench.Repairs.RopeTransform.IndexCSE
