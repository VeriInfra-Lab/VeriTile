import bench.tritonbench_g.chunk_delta_fwd.Proofs.SourceSafetyPrefix
import bench.tritonbench_g.chunk_delta_fwd.Proofs.SourceSafetyBlockPointer

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety ChunkDeltaFwd
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem entry_tail (M : Algebra α) (p : Parameters) (s : State α)
    (hk : s.pids 0 = 0) (bounds : RegionName → Nat)
    (hb : Bounds p (s.pids 1,s.pids 2) bounds) :
    traceSafeList M bounds ((outer p).before.drop 4) (entryStart M p s) := by
  simp [outer,kernel,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
  rw [singleton]
  cases hp : p.useSeed
  · simp [stepSafe_unfold,safeOp_unfold,evalOp_unfold,hp]
  · simp [stepSafe_unfold,safeOp_unfold,evalOp_unfold,hp]
    apply traceSafeList_cons_intro
    · simp [stepSafe_unfold,safeExpr_unfold,safeOp_unfold]
    · intro next ran
      simp [step,evalExpr,evalOp_unfold,evalOffsets,Option.bind,numeric,bop,
        entryStart,Region.cast,hk] at ran
      cases ran
      rw [singleton]
      simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
        safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
        evalOp_unfold,Option.bind,entryStart,Region.cast,hk,
        BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
      intro k v hk' hv
      exact hb.seed hp k.val (s.pids 1*p.BV+v.val) hk' hv

theorem entry (M : Algebra α) (p : Parameters) (s : State α)
    (hk : s.pids 0 = 0) (bounds : RegionName → Nat)
    (hb : Bounds p (s.pids 1,s.pids 2) bounds) :
    traceSafeList M bounds (outer p).before s := by
  rw [← List.take_append_drop 4 (outer p).before,traceSafeList_append]
  refine ⟨entry_prefix_safe M p s bounds,?_⟩
  intro next ran
  rw [entry_prefix_run] at ran
  cases ran
  exact entry_tail M p s hk bounds hb

theorem history (M : Algebra α) (p : Parameters)
    (pid : PID) (input : Memory α) (current : State α) (c : Nat)
    (frozen : HValues p α) (ctx : Ready pid input current) (hh : HasH p frozen current)
    (bounds : RegionName → Nat) (hb : Bounds p pid bounds) (hc : c < p.NT) :
    traceSafeList M bounds (inner p).before
      (current.setReg "i_t" .nat [] (fun _ => c)) := by
  simp [ChunkDeltaFwd.inner,outer,kernel,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
  apply traceSafeList_cons_intro
  · simp [stepSafe_unfold,safeExpr_unfold,safeOp_unfold]
  · intro next ran
    simp [step,evalExpr,evalOp_unfold,evalOffsets,Option.bind,numeric,bop,Region.cast,
      ctx.ik,ctx.iv,ctx.bh] at ran
    cases ran
    apply traceSafeList_cons_intro
    · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
        safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
        evalOp_unfold,Option.bind,Region.cast,
        BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
      intro k v hk hv
      exact hb.history c k.val (pid.1*p.BV+v.val) hc hk hv
    · intro last _
      rw [singleton]
      simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold]

theorem finish (M : Algebra α) (p : Parameters)
    (pid : PID) (input : Memory α) (s : State α) (frozen : HValues p α)
    (ctx : Ready pid input s) (hh : HasH p frozen s)
    (bounds : RegionName → Nat) (hb : Bounds p pid bounds) :
    traceSafeList M bounds (outer p).after s := by
  simp [outer,kernel,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
  rw [singleton]
  cases hp : p.storeFinal
  · simp [stepSafe_unfold,safeOp_unfold,evalOp_unfold,hp]
  · simp [stepSafe_unfold,safeOp_unfold,evalOp_unfold,hp]
    apply traceSafeList_cons_intro
    · simp [stepSafe_unfold,safeExpr_unfold,safeOp_unfold]
    · intro next ran
      simp [step,evalExpr,evalOp_unfold,evalOffsets,Option.bind,numeric,bop,Region.cast,
        ctx.ik,ctx.iv,ctx.bh] at ran
      cases ran
      rw [singleton]
      simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
        safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
        evalOp_unfold,Option.bind,Region.cast,
        BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
      intro k v hk hv
      exact hb.final hp k.val (pid.1*p.BV+v.val) hk hv

theorem update (M : Algebra α) (p : Parameters) (s : State α) (bounds : RegionName → Nat) :
    traceSafeList M bounds (inner p).after s := by
  simp [ChunkDeltaFwd.inner,outer,kernel,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
  rw [singleton]
  simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold]

end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.SourceSafety
