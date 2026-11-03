import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Mathematical_RealLoop
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Mathematical_RealAccess
import VeriTile.Triton.Memory.TraceProofs
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.SignedLoop

/-! Address safety follows the same full dynamic-loop invariant as execution.
The hypotheses bound initial-array windows, not prepared pointers or a supplied
loop run. Both input windows stop at the actual shortened chunk length. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealLoopSafety
open VeriTile.Triton
open VeriTile.Bench.Repairs.BmmChunkFwd.ControlArithmetic
attribute [local simp 2000] VeriTile.Triton.eval_truncDiv VeriTile.Triton.eval_truncMod
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

structure Bounds (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) (bounds : RegionBounds) : Prop where
  a : ∀ (i : Fin BM) (k : Nat), RealSetup.rows s BM BN p i < RealSetup.limit s p → k < p.K →
    RealSetup.aBase s p + (RealSetup.rows s BM BN p i * p.aRow + k * p.aColumn) < bounds p.A
  b : ∀ (k : Nat) (j : Fin BN), k < p.K → RealSetup.columns s BN p j < RealSetup.limit s p →
    RealSetup.bBase s p + (k * p.bColumn + RealSetup.columns s BN p j * p.bRow) < bounds p.B

theorem iteration_safe (swapped : Bool) (initial s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (bounds : RegionBounds) (hb : Bounds initial BM BN p bounds)
    (c : Nat) (h : RealLoop.Holds initial BM BN BK p c s) :
    Stmt.TraceSafeList bounds (RealLoads.code swapped BM BN BK p) (s.setReg "k" .int [] (Tile.scalar (c : Int))) := by
  let entry := s.setReg "k" .int [] (Tile.scalar (c : Int))
  have hr : entry.regs .int [BM] "offs_m" = some (Tile.vec (fun i => (RealSetup.rows initial BM BN p i : Int))) := by
    simpa [entry, RealSetup.result] using h.fixed .int [BM] "offs_m" (by decide)
  have hn : entry.regs .int [BN] "offs_n" = some (Tile.vec (fun i => (RealSetup.columns initial BN p i : Int))) := by
    simpa [entry, RealSetup.result] using h.fixed .int [BN] "offs_n" (by decide)
  have hk : entry.regs .int [BK] "offs_k" = some (Tile.vec (fun i => (i.val : Int))) := by
    simpa [entry, RealSetup.result] using h.fixed .int [BK] "offs_k" (by decide)
  have hl : RealLoads.LimitAt entry (RealSetup.limit initial p) := by
    refine ⟨RealSetup.signedLimit initial p, ?_, ?_⟩
    · simpa [entry,RealSetup.result] using h.fixed .int [] "chunk_size_limit" (by decide)
    · simp only [RealSetup.signedLimit,RealSetup.limit,←Int.natCast_mul,min_def]
      split_ifs <;> omega
  have hc : entry.regs .int [] "k" = some (Tile.scalar (c : Int)) := by simp [entry]
  have hd : entry.regs .ptr [BM, BK] "a_ptrs" = some (RealLoop.aPointers initial BM BN BK p c) := by
    simpa [entry] using h.a
  have ha : entry.regs .ptr [BK, BN] "b_ptrs" = some (RealLoop.bPointers initial BN BK p c) := by
    simpa [entry] using h.b
  have hsafe := RealAccess.a_safe bounds entry BM BK p.K (RealSetup.limit initial p) c
    (RealSetup.rows initial BM BN p) (RealLoop.aPointers initial BM BN BK p c) hd hr hk hl hc
    (fun i k hi hk => hb.a i (c * BK + k.val) hi ((BmmChunkBwd.RealMath.tail_mask _ _ _ _).mp hk))
  rw [RealLoads.code]
  apply TraceProofs.assign_cons hsafe
  intro v
  apply TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def])
  intro _
  apply TraceProofs.assign_cons
  · apply RealAccess.b_safe bounds _ BN BK p.K (RealSetup.limit initial p) c
      (RealSetup.columns initial BN p) (RealLoop.bPointers initial BN BK p c)
    · simpa using ha
    · simpa using hn
    · simpa using hk
    · obtain ⟨raw,hr,he⟩ := hl
      exact ⟨raw,by simpa using hr,he⟩
    · simpa using hc
    · intro k j hk hj
      exact hb.b (c * BK + k.val) j ((BmmChunkBwd.RealMath.tail_mask _ _ _ _).mp hk) hj
  · intro _
    apply TraceProofs.all_safe
    simp only [List.forall_mem_cons, List.not_mem_nil, false_implies, forall_const, and_true]
    cases swapped <;> simp only [RealLoads.update, Bool.false_eq_true, ↓reduceIte, Stmt.TraceSafe, Op.SafeAt.eq_def, and_self, implies_true]

theorem loop_safe (swapped : Bool) (initial : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (bounds : RegionBounds) (hb : Bounds initial BM BN p bounds) :
    Stmt.TraceSafe bounds (RealLoop.statement swapped BM BN BK p) (RealSetup.result initial BM BN BK p) := by
  rw [RealLoop.statement, Stmt.TraceSafe]
  refine ⟨by simp [Op.SafeAt.eq_def],
    by simp [RealProgram.loopBound,Op.SafeAt.eq_def],by simp [Op.SafeAt.eq_def],?_⟩
  have stop_value : evalOp (RealProgram.loopBound p.K BK) (RealSetup.result initial BM BN BK p)=
      some (Tile.scalar (BmmChunkBwd.RealMath.blocks p.K BK : Int)) := by
    simp [RealProgram.loopBound,evalOp.eq_def,Tile.bop,Tile.scalar,
      NumericDType.add,NumericDType.sub,bind,Option.bind,BmmChunkBwd.RealMath.blocks,-Tile.scalar_eta]
  rw [stop_value]
  simp only [evalOp.eq_def,Tile.scalar]
  change Stmt.forRangeIntTraceSafe bounds "k" (0 : Int) (BmmChunkBwd.RealMath.blocks p.K BK : Int) 1
    (RealLoads.code swapped BM BN BK p) (RealSetup.result initial BM BN BK p)
  apply BmmChunkBwd.SignedLoop.safe (P := RealLoop.Holds initial BM BN BK p) (c := 0)
  · intro c s _ h
    exact iteration_safe swapped initial s BM BN BK p bounds hb c h
  · intro c s t hc h he
    obtain ⟨u, hu, hf⟩ := RealLoop.iteration swapped initial BM BN BK p c s (by simpa using hc) h
    rw [he] at hu
    cases hu
    exact hf
  · exact RealLoop.initial initial BM BN BK p

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealLoopSafety
