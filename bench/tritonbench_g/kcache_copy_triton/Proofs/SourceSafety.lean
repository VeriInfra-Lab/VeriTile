import bench.tritonbench_g.kcache_copy_triton.Proofs.KernelFP
import VeriTile.Triton.Float.StructuralSafety
namespace VeriTile.Bench.Repairs.KcacheCopyTriton.KernelSafety
open VeriTile.Triton FP.Structural Source KernelFP
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

private theorem head_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (headBody p) s := by
  simp [headBody,kernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,bop,Region.cast,Option.bind,
    FP.Structural.eval_truncDiv,FP.Structural.eval_truncMod,headState,KernelFP.sequence,
    Broadcast.leftIndex,Broadcast.rightIndex]

private theorem metadata_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α)
    (lengths : (KernelFP.sequence p s.pids).toNat<bounds p.lengths.cast)
    (block : (KernelFP.sequence p s.pids*p.strideBts).toNat+
      ((position p s).tdiv p.blockSize*p.strideBtb).toNat<bounds p.blockTables.cast) :
    Safety.traceSafeList M bounds (metadataBody p) (headState p s) := by
  simp [metadataBody,kernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,bop,Region.cast,Option.bind,
    FP.Structural.eval_truncDiv,FP.Structural.eval_truncMod,metadataState,headState,
    selectedBlock,position,KernelFP.sequence,Broadcast.leftIndex,Broadcast.rightIndex,
    lengths,block]
  simpa [position,KernelFP.sequence,Region.cast,integralFloorDiv,←IntegerExpr.tdiv_formula,Int.tmod_def,
    Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right] using And.intro lengths block

private theorem tail_safe {α : Type} [Inhabited α] (M : Algebra α) (v : Bool)
    (p : Parameters) (bounds : RegionName → Nat) (s : State α)
    (input : ∀ i : TileIndex [p.width],sourceOffset p s.pids i<bounds p.K)
    (output : ∀ i : TileIndex [p.width],targetOffset p s i<bounds p.KCache) :
    Safety.traceSafeList M bounds (tailBody v p) (metadataState p s) := by
  cases v <;> simp [tailBody,kernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,store,numeric,bop,ComputeDType.eraseDType,
    Region.cast,Option.bind,preparedState,metadataState,headState,
    sourceOffset,targetOffset,Broadcast.leftIndex,Broadcast.rightIndex,
    Int.add_comm,Int.add_left_comm,Int.add_assoc]
  all_goals
    simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
    constructor
    · intro i
      simpa [sourceOffset,Region.cast,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc] using input (i,())
    · intro i
      have h := output (i,())
      unfold targetOffset at h
      convert h using 1
      apply congrArg Int.toNat
      push_cast
      ring

/-- Actual metadata gather, FP32 data load and destination scatter safety.
Both sources retain the complete signed metadata and arbitrary alias scope. -/
theorem trace_safe {α : Type} [Inhabited α] (M : Algebra α) (v : Bool)
    (p : Parameters) (bounds : RegionName → Nat) (s : State α)
    (lengths : (KernelFP.sequence p s.pids).toNat<bounds p.lengths.cast)
    (block : (KernelFP.sequence p s.pids*p.strideBts).toNat+
      ((position p s).tdiv p.blockSize*p.strideBtb).toNat<bounds p.blockTables.cast)
    (input : ∀ i : TileIndex [p.width],sourceOffset p s.pids i<bounds p.K)
    (output : ∀ i : TileIndex [p.width],targetOffset p s i<bounds p.KCache) :
    TraceSafe M bounds (kernel v p) s := by
  unfold TraceSafe
  rw [decomposition]
  change Safety.traceSafeList M bounds (headBody p++metadataBody p++tailBody v p) s
  rw [List.append_assoc,Safety.traceSafeList_append]
  refine ⟨head_safe M p bounds s,?_⟩
  intro t run
  rw [head_run] at run
  cases run
  rw [Safety.traceSafeList_append]
  refine ⟨metadata_safe M p bounds s lengths block,?_⟩
  intro t run
  rw [metadata_run] at run
  cases run
  exact tail_safe M v p bounds s input output

#axiomsClean trace_safe
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.KernelSafety
