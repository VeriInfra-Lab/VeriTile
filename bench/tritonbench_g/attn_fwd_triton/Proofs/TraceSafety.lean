import bench.tritonbench_g.attn_fwd_triton.Proofs.SafetyMiddle
import bench.tritonbench_g.attn_fwd_triton.Proofs.ViewBounds
import bench.tritonbench_g.attn_fwd_triton.Proofs.AttnFwdTriton_Verified_Execution_Execution

/-! The complete source trace is bounded from ordinary logical-view extents,
sufficient packed scales and an in-grid program. Both source stages remain. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.TraceSafety
open VeriTile.Triton Memory
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccess
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem epilogue_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (BM BD : Nat) (reciprocal : Bool)
    (hp : s.regs .ptr [BM,BD] "O_block_ptr" = some (outputPointer c BM BD))
    (hr : s.regs .int [BM] "offs_m" = some ⟨fun i => c.startM*BM+i.1.val⟩)
    (hctx : s.regs .int [] "n_ctx" = some (Tile.scalar c.p.context))
    (hb : AccessSafety.OutputBound bounds c BM BD) :
    Stmt.TraceSafeList bounds (epilogue BM BD c.p.headDim reciprocal) s := by
  cases reciprocal
  all_goals
    simp only [epilogue,VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSource.normalization,
      Bool.false_eq_true,Bool.true_eq,↓reduceIte]
    apply TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def])
    intro tile
    apply Stmt.TraceSafeList.cons_intro
    · exact AccessSafety.output_safe bounds _ c BM BD (by simp [hp]) (by simp [hr]) (by simp [hctx]) hb
    · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem safe (bounds : RegionBounds) (s : BlockState) (p : Parameters) (BD : Nat) (reciprocal : Bool)
    (hv : ViewBounds.Windows bounds p) (hp : ViewBounds.Program (Coordinates.configuration s p)) :
    Stmt.TraceSafeList bounds (Verified.code 128 64 BD p reciprocal) s := by
  let c := Coordinates.configuration s p
  let diagonal := decide (p.stage=2)
  have hb1 : SafetyLoop.Bounds bounds c 64 BD (FirstSetup.firstLower c.startM 128 diagonal)
      (FirstSetup.firstUpper c.startM 128 p.context diagonal) := by
    apply ViewBounds.loop_bounds bounds c BD _ _ hv hp
    cases hd : diagonal
    · exact Nat.le_of_lt hp.row
    · exact Nat.min_le_right _ _
  have hb2 := ViewBounds.loop_bounds bounds c BD (c.startM*128) (min ((c.startM+1)*128) p.context) hv hp (Nat.min_le_right _ _)
  obtain ⟨sp,hpstep,hfixed,hcarry,hcaller,hout,hpf⟩ := Prologue.executes s p 128 64 BD
  have hpsafe := SafetyPrologue.safe bounds s p 128 64 BD
    (ViewBounds.query bounds c BD hv hp) (ViewBounds.query_scale bounds c hv hp)
  have hlsafe := SafetyMiddle.safe bounds s sp c 128 64 BD diagonal (Initialization.initial 128 BD)
    (by omega) hfixed hcarry hcaller hb1 hb2
  obtain ⟨sl,hlstep,hlfixed,hlcarry,hlf⟩ := TwoLoops.executes s sp c 128 64 BD diagonal (Initialization.initial 128 BD)
    (by omega) hfixed hcarry hcaller
  have hop : sl.regs .ptr [128,BD] "O_block_ptr" = some (outputPointer c 128 BD) :=
    (hlf.regs _ _ "O_block_ptr" (by decide)).trans hout
  rw [Execution.body,List.append_assoc]
  apply append_of_run hpsafe hpstep
  apply append_of_run hlsafe hlstep
  exact epilogue_safe bounds sl c 128 BD reciprocal hop hlfixed.rows hlfixed.context
    (ViewBounds.output bounds c BD hv hp)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.TraceSafety
