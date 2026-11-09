import bench.tritonbench_g.bmm_chunk_bwd.Proofs.Mathematical_RealLoop
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.Mathematical_RealAccess
import VeriTile.Triton.Memory.TraceProofs

/-! Address safety follows the same full dynamic-loop invariant as execution.
The hypotheses bound initial-array windows, not prepared pointers or a supplied
loop run. They include the source's Dout rows up to chunk_size. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealLoopSafety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

structure Bounds (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) (bounds : RegionBounds) : Prop where
  dout : ∀ (i : Fin BM) (k : Nat), RealSetup.rows s BM BN p i < p.chunk → k < RealSetup.limit s p →
    RealSetup.dBase s p + (RealSetup.rows s BM BN p i * p.dColumn + k * p.dRow) < bounds p.Dout
  a : ∀ (k : Nat) (j : Fin BN), k < RealSetup.limit s p → RealSetup.columns s BN p j < p.K →
    RealSetup.aBase s p + (k * p.aRow + RealSetup.columns s BN p j * p.aColumn) < bounds p.A

theorem iteration_safe (initial s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (bounds : RegionBounds) (hb : Bounds initial BM BN p bounds)
    (c : Nat) (h : RealLoop.Holds initial BM BN BCS p c s) :
    Stmt.TraceSafeList bounds (RealLoads.code BM BN BCS p) (s.setReg "cs" .int [] (Tile.scalar (c : Int))) := by
  let entry := s.setReg "cs" .int [] (Tile.scalar (c : Int))
  have hr : entry.regs .int [BM] "offs_m" = some (Tile.vec fun i => (RealSetup.rows initial BM BN p i : Int)) := by
    simpa [entry, RealSetup.result] using h.fixed .int [BM] "offs_m" (by decide)
  have hn : entry.regs .int [BN] "offs_n" = some (Tile.vec fun j => (RealSetup.columns initial BN p j : Int)) := by
    simpa [entry, RealSetup.result] using h.fixed .int [BN] "offs_n" (by decide)
  have hk : entry.regs .int [BCS] "offs_cs" = some (Tile.vec fun k => (k.val : Int)) := by
    simpa [entry, RealSetup.result] using h.fixed .int [BCS] "offs_cs" (by decide)
  have hl : entry.regs .int [] "chunk_size_limit" = some (Tile.scalar (RealSetup.limitInt initial p)) := by
    simpa [entry, RealSetup.result] using h.fixed .int [] "chunk_size_limit" (by decide)
  have hc : entry.regs .int [] "cs" = some (Tile.scalar (c : Int)) := by simp [entry]
  have hd : entry.regs .ptr [BM, BCS] "dout_ptrs" = some (RealLoop.dPointers initial BM BN BCS p c) := by
    simpa [entry] using h.dout
  have ha : entry.regs .ptr [BCS, BN] "a_ptrs" = some (RealLoop.aPointers initial BN BCS p c) := by
    simpa [entry] using h.a
  have hsafe := RealAccess.dout_safe bounds entry BM BCS p.chunk (RealSetup.limitInt initial p) c
    (RealSetup.rows initial BM BN p) (RealLoop.dPointers initial BM BN BCS p c) hd hr hk hl hc
    (fun i k hi hk => hb.dout i (c * BCS + k.val) hi ((RealMath.tail_mask _ _ _ _).mp (by simpa only [RealSetup.limitInt_toNat] using hk)))
  rw [RealLoads.code]
  apply TraceProofs.assign_cons hsafe
  intro v
  apply TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def])
  intro _
  apply TraceProofs.assign_cons
  · apply RealAccess.a_safe bounds _ BN BCS p.K (RealSetup.limitInt initial p) c
      (RealSetup.columns initial BN p) (RealLoop.aPointers initial BN BCS p c)
    · simpa using ha
    · simpa using hn
    · simpa using hk
    · simpa using hl
    · simpa using hc
    · intro k j hk hj
      exact hb.a (c * BCS + k.val) j ((RealMath.tail_mask _ _ _ _).mp (by simpa only [RealSetup.limitInt_toNat] using hk)) hj
  · intro _
    apply TraceProofs.all_safe
    simp only [List.forall_mem_cons, List.not_mem_nil, false_implies, forall_const, and_true]
    simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, and_self, implies_true]

theorem loop_safe (initial : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (bounds : RegionBounds) (hb : Bounds initial BM BN p bounds) :
    Stmt.TraceSafe bounds (RealLoop.statement BM BN BCS p) (RealSetup.result initial BM BN BCS p) := by
  rw [RealLoop.statement, Stmt.TraceSafe]
  refine ⟨by simp [Op.SafeAt.eq_def],
    by simp [RealProgram.loopBound, Op.SafeAt.eq_def], by simp [Op.SafeAt.eq_def], ?_⟩
  rw [RealLoop.bound_eval initial BM BN BCS p]
  simp only [evalOp.eq_def, Tile.scalar]
  change Stmt.forRangeIntTraceSafe bounds "cs" (0 : Int) (RealLoop.stop initial BCS p) 1
    (RealLoads.code BM BN BCS p) (RealSetup.result initial BM BN BCS p)
  apply SignedLoop.safe (P := RealLoop.Holds initial BM BN BCS p) (c := 0)
  · intro c s _ h
    exact iteration_safe initial s BM BN BCS p bounds hb c h
  · intro c s t hc h he
    obtain ⟨u, hu, hf⟩ := RealLoop.iteration initial BM BN BCS p c s
      (by simpa only [RealLoop.stop_count] using hc) h
    rw [he] at hu
    cases hu
    exact hf
  · exact RealLoop.initial initial BM BN BCS p

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealLoopSafety
