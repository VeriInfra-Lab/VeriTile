import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyPrelude
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyLoop
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyOutput

/-! Complete real-source trace safety from actual active-address bounds. The
query and normalizer ranges use the proved trip count, while K is loaded once
even if the query range is empty. Output allocation is required only for the
source store mask. No numerical equation is a safety premise. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealSafety
open VeriTile.Triton
open VeriTile.Triton.Memory.AccessHelpers
set_option maxHeartbeats 1000000
noncomputable section

structure Bounds (bounds : RegionBounds) (c : RealLoop.Config) (stop : Nat)
    (Out : RegionName) (baseOut : Nat) : Prop where
  query : ∀ j, j < RealLoop.tripCount stop c.B →
    BlockBounds bounds (RealLoads.queryPointer c.Q c.B c.D c.N c.baseQ c.sqm c.sqk (j*c.B))
      (RealLoads.checks c.evenM)
  key : BlockBounds bounds (RealLoads.keyPointer c.K c.B c.D c.NK c.baseK c.skn c.skk c.startN)
    (RealLoads.checks c.evenN)
  normalizer : ∀ row, row < c.N → c.baseM+row < bounds c.M
  output : ∀ i : Fin c.B, c.startN*c.B+i.val < c.NK → baseOut+c.startN*c.B+i.val < bounds Out

theorem correct (bounds : RegionBounds) (s : BlockState) (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh H HK N RN NK B D : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN candidate : Bool)
    (hB : 0 < B) (keyOffsetBound : s.pids 0 * B < 2147483648)
    (hb : Bounds bounds (RealPrelude.config s Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk
      offset size scale window complement evenM evenN candidate) N Out
      (s.pids 1 / H * soz + s.pids 1 % H * soh)) :
    Stmt.TraceSafeList bounds (RealProgram.program Q K M Out sqz sqh sqm sqk skz skh skn skk soz soh
      H HK N RN NK B D offset size scale window complement evenM evenN candidate) s := by
  let c := RealPrelude.config s Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk
    offset size scale window complement evenM evenN candidate
  obtain ⟨sp,hp,ip,hlo,hhi,hz,hh,_,_⟩ := RealPrelude.correct s Q K M H HK N RN NK B D
    sqz sqh sqm sqk skz skh skn skk offset size scale window complement evenM evenN candidate keyOffsetBound
  obtain ⟨sl,hl,il,fl⟩ := RealLoop.correct c s sp N hB ip
  have hloop : stepStmt (.forRangeInt "start_m" (.ref .int [] "lo") (.ref .int [] "hi")
      (.constNat B) c.body) sp = some sl := by
    simp only [stepForRangeIntAux.forRangeInt_unfold, evalOp_ref, hlo, hhi, evalOp_constNat,
      Option.bind_some, Tile.scalar_data]
    exact hl
  have safeLoop : Stmt.TraceSafe bounds (.forRangeInt "start_m" (.ref .int [] "lo")
      (.ref .int [] "hi") (.constNat B) c.body) sp := by
    simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp_ref, hlo, hhi, evalOp_constNat,
      Tile.scalar_data, true_and]
    exact RealSafetyLoop.correct bounds c s sp N hB ip hb.query hb.normalizer
  have hzL := (fl.regs .int [] "off_z" (by simp [RealIteration.writes])).trans hz
  have hhL := (fl.regs .int [] "off_h" (by simp [RealIteration.writes])).trans hh
  simp only [RealProgram.program, List.append_assoc, List.singleton_append]
  apply append_of_run (RealSafetyPrelude.correct bounds s Q K M H HK N RN NK B D
    sqz sqh sqm sqk skz skh skn skk scale evenN keyOffsetBound hb.key) hp
  apply cons_of_run safeLoop hloop
  exact RealSafetyOutput.correct bounds sl Out B NK soz soh (s.pids 1 / H) (s.pids 1 % H)
    (s.pids 0) hzL hhL il.startN hb.output

theorem source_safe (bounds : RegionBounds) (s : BlockState) (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN candidate : Bool)
    (hBlock : BM = BN) (hB : 0 < BN) (keyOffsetBound : s.pids 0 * BN < 2147483648)
    (hb : Bounds bounds (RealPrelude.config s Q K M H HK N RN NK BN BD sqz sqh sqm sqk skz skh skn skk
      offset size scale window complement evenM evenN candidate) N Out
      (s.pids 1 / H * soz + s.pids 1 % H * soh)) :
    let source := if candidate then TypedSource.scalarLeft Q K M Out
        sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
        offset size scale window complement evenM evenN hBlock
      else TypedSource.original Q K M Out
        sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
        offset size scale window complement evenM evenN hBlock
    source.eraseDType.toAlgKernel.TraceSafe bounds s := by
  cases candidate
  all_goals
    dsimp only
    simp only [Bool.false_eq_true, ↓reduceIte, Kernel.TraceSafe,
      RealProgram.original_source, RealProgram.candidate_source]
    exact correct bounds s Q K M Out sqz sqh sqm sqk skz skh skn skk soz soh H HK N RN NK BN BD
      offset size scale window complement evenM evenN _ hB keyOffsetBound hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealSafety
