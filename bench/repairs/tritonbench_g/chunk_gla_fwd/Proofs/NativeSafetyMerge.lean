import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyBasics

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite FPMerge
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem merge_prologue (M : Algebra α) (p : Layout) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (engine M) bounds (branch p).before s := by
  apply pure_trace
  intro stmt listed current
  simp [branch,kernel,TypedMerge.original,ComputeKernel.surfaceBody,
    Source.conditional?,Source.prefixBefore] at listed
  rcases listed with rfl|rfl|rfl <;>
    simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem merge_entry (M : Algebra α) (p : Layout) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (engine M) bounds (parts p).before s := by
  apply pure_trace
  intro stmt listed current
  simp [parts,branch,kernel,TypedMerge.original,ComputeKernel.surfaceBody,
    Source.conditional?,Source.dynamicRange?,Source.prefixBefore] at listed
  rcases listed with rfl|rfl <;>
    simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem merge_body (M : Algebra α) (p : Layout) (bounds : RegionName → Nat)
    (initial current : State α) (f : Nat) (hf : f < p.NK)
    (ctx : Context p initial current) (capacity : MergeCapacity p initial bounds) :
    Safety.traceSafeList (engine M) bounds (parts p).body
      (current.setReg "i_k" .nat [] (fun _ => f)) := by
  simp [parts,branch,kernel,TypedMerge.original,ComputeKernel.surfaceBody,
    Source.conditional?,Source.dynamicRange?,Source.prefixBefore]
  simp only [Safety.traceSafeList_cons,singleton]
  simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
  simp [Safety.safeOp_unfold,Safety.addressSafe,Safety.maskActive,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    ctx.it,ctx.ic,ctx.bh,ctx.nh,BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,
    TileShape.indexToList]
  intro i j active
  exact capacity.input f hf (i,j,PUnit.unit) active

#axiomsClean merge_body
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
