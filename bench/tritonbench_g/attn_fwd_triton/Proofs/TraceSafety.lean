import bench.tritonbench_g.attn_fwd_triton.Proofs.SafetyPrologue
import bench.tritonbench_g.attn_fwd_triton.Proofs.ViewBounds
import bench.tritonbench_g.attn_fwd_triton.Proofs.Output

/-! Output-fragment trace safety. Complete source safety is established
through the actual captured helper invocations in FlatTraceSafety. -/
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
    simp only [epilogue,VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealSource.normalization,
      Bool.false_eq_true,Bool.true_eq,↓reduceIte]
    apply TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def])
    intro tile
    apply Stmt.TraceSafeList.cons_intro
    · exact AccessSafety.output_safe bounds _ c BM BD (by simp [hp]) (by simp [hr]) (by simp [hctx]) hb
    · intro _ _; exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.TraceSafety
