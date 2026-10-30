import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.DirectReal
import VeriTile.Triton.Memory.RegionalScheduledKernelIO

/-! Safety of the selected explicit-pointer source, at its actual intermediate
states. The older block-pointer source safety theorem is not used here. -/
namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.IOSourceSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math Real
open VeriTile.Bench.Repairs.ChunkGlaSimple Memory
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputSafety (MatrixBounds)
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

private theorem cons_of_run {bounds : RegionBounds} {st : Stmt} {rest : List Stmt}
    {s t : BlockState} (he : stepStmt st s = some t) :
    Stmt.TraceSafeListR .triv bounds (st :: rest) s ↔
      st.TraceSafeR .triv bounds s ∧ Stmt.TraceSafeListR .triv bounds rest t := by
  simp only [Stmt.TraceSafeListR, stepStmtR_triv, he]

private theorem assign_cons {d sh name} {e : Op d sh} {s : BlockState} {bounds rest}
    (safe : e.SafeAtR .triv bounds s)
    (next : ∀ v, Stmt.TraceSafeListR .triv bounds rest (s.setReg name d sh v)) :
    Stmt.TraceSafeListR .triv bounds (.assign d sh name e :: rest) s := by
  apply Stmt.TraceSafeListR.cons_intro (by simpa only [Stmt.TraceSafeR] using safe)
  intro t he
  simp only [stepStmtR_triv, stepStmt] at he
  cases h : evalOp e s with
  | none => simp [h] at he
  | some v =>
    simp only [h, bind, Option.bind_some, Option.some.injEq] at he
    exact he ▸ next v

theorem load2_safe (s : BlockState) (r : RegionName) (ptr x y : RegName)
    (m n rows cols b : Nat) (xs : Fin m → Nat) (ys : Fin n → Nat)
    (ax : Fin m → Nat) (ay : Fin n → Nat) (bounds : RegionBounds)
    (hb : ∀ i j, xs i < rows ∧ ys j < cols → b + ax i + ay j < bounds r)
    (hp : s.regs .ptr [m,n] ptr = some (matrix r b ax ay))
    (hx : s.regs .nat [m] x = some (Tile.vec xs))
    (hy : s.regs .nat [n] y = some (Tile.vec ys)) :
    (Program.load2 ptr x y m n rows cols).SafeAtR .triv bounds s := by
  rw [Program.load2, Op.SafeAtR, Op.SafeAtR.eq_def]
  refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Program.mask2, Program.col, Program.row, Op.SafeAtR, Op.SafeAtR.eq_def], ?_⟩
  intro ps he i active
  simp only [evalOpR_triv, evalOp_ref, hp, Option.some.injEq] at he
  subst ps
  obtain ⟨ms, hm, hi⟩ := active
  rw [evalOpR_triv, mask2_eval s x y m n rows cols xs ys hx hy] at hm
  cases hm
  exact hb i.1 i.2.1 (of_decide_eq_true hi)

theorem gate_safe (p : Layout) (l : Launch) (r : Regions) (s : BlockState)
    (bounds : RegionBounds)
    (hb : ∀ i : Fin p.BT, Math.row p l i.val < p.T → gateAddress p l i.val < bounds r.gate)
    (hp : s.regs .ptr [p.BT] "p_g" = some (gatePointer p l r))
    (hr : s.regs .nat [p.BT] "rows" = some (vector p.BT (l.chunk*p.BT))) :
    (Program.gateLoad p).SafeAtR .triv bounds s := by
  rw [Program.gateLoad, Op.SafeAtR, Op.SafeAtR.eq_def]
  refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR, Op.SafeAtR.eq_def], ?_⟩
  intro ps he i active
  simp only [evalOpR_triv, evalOp_ref, hp, Option.some.injEq] at he
  subst ps
  obtain ⟨ms, hm, hi⟩ := active
  simp only [evalOpR_triv, evalOp, hr, Option.bind_some, Option.bind_eq_bind] at hm
  cases hm
  exact hb i.1 (of_decide_eq_true hi)

theorem store_safe (p : Layout) (l : Launch) (r : Regions) (s : BlockState)
    (bounds : RegionBounds) (hb : Safe.TailBounds p l r bounds)
    (hp : s.regs .ptr [p.BT,p.BV] "p_o" = some (valuePointer p l r.output))
    (hr : s.regs .nat [p.BT] "rows" = some (vector p.BT (l.chunk*p.BT)))
    (hc : s.regs .nat [p.BV] "cols" = some (vector p.BV (l.value*p.BV))) :
    (Program.store p).TraceSafeR .triv bounds s := by
  rw [Program.store, Stmt.TraceSafeR]
  refine ⟨by simp [MemAccess.SafeAtR, Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR, Program.mask2,
    Program.col, Program.row, Op.SafeAtR, Op.SafeAtR.eq_def], ?_⟩
  intro ps he i active
  simp only [evalOpR_triv, evalOp_ref, hp, Option.some.injEq] at he
  subst ps
  obtain ⟨ms, hm, hi⟩ := active
  rw [evalOpR_triv, mask2_eval s "rows" "cols" p.BT p.BV p.T p.V _ _ hr hc] at hm
  cases hm
  simpa only [valuePointer, matrix, Nat.mul_one] using
    hb.output i.1 i.2.1 (of_decide_eq_true hi)

theorem iteration_safe (p : Layout) (l : Launch) (r : Regions) (kb : Nat)
    (bounds : RegionBounds) (hb : Safe.StepBounds p l r kb bounds) (s : BlockState)
    (hh : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : s.regs .nat [] "i_t" = some (Tile.scalar l.chunk))
    (hr : s.regs .nat [p.BT] "rows" = some (vector p.BT (l.chunk*p.BT)))
    (hc : s.regs .nat [p.BV] "cols" = some (vector p.BV (l.value*p.BV)))
    (hk : s.regs .nat [] "i_k" = some (Tile.scalar kb)) :
    Stmt.TraceSafeListR .triv bounds (Program.iteration p r) s := by
  unfold Program.iteration Real.iteration chunkGlaLoopBody
  simp only [List.drop_succ_cons, List.drop_zero, List.cons_append, List.nil_append]
  rw [cons_of_run (stepStmt_assign_eq_some (indices_eval s "i_k" p.BK kb hk))]
  refine ⟨by simp [Stmt.TraceSafeR, Program.indices, Program.times, Op.SafeAtR, Op.SafeAtR.eq_def], ?_⟩
  rw [cons_of_run (stepStmt_assign_eq_some
    (queryPtr_eval p l r kb _ (by simp [hh]) (by simp [hr, vector]) (by simp [vector])))]
  refine ⟨by simp [Stmt.TraceSafeR, Program.queryPtr, Program.ptr2, Program.times,
    Program.scaleCol, Program.col, Program.row, Op.SafeAtR, Op.SafeAtR.eq_def], ?_⟩
  rw [cons_of_run (stepStmt_assign_eq_some
    (keyPtr_eval p l r kb _ (by simp [hh]) (by simp [hr, vector]) (by simp [vector])))]
  refine ⟨by simp [Stmt.TraceSafeR, Program.keyPtr, Program.ptr2, Program.times,
    Program.scaleRow, Program.col, Program.row, Op.SafeAtR, Op.SafeAtR.eq_def], ?_⟩
  rw [cons_of_run (stepStmt_assign_eq_some
    (historyPtr_eval p l r kb _ (by simp [hh]) (by simp [ht]) (by simp [hc]) (by simp)))]
  refine ⟨by simp [Stmt.TraceSafeR, Program.historyPtr, Program.ptr2, Program.historyBase,
    Program.times, Program.scaleCol, Program.col, Program.row, Op.SafeAtR, Op.SafeAtR.eq_def], ?_⟩
  apply assign_cons (load2_safe _ r.query "p_q" "rows" "keys" _ _ _ _ _ _ _ _ _ bounds
    (by simpa only [Nat.mul_one] using hb.query) (by simp [queryPointer]) (by simp [hr, vector]) (by simp [vector]))
  intro q
  apply assign_cons (load2_safe _ r.key "p_k" "keys" "rows" _ _ _ _ _ _ _ _ _ bounds
    (by simpa only [Nat.mul_one] using hb.key) (by simp [keyPointer]) (by simp [vector]) (by simp [hr, vector]))
  intro k
  apply assign_cons (load2_safe _ r.history "p_h" "keys" "cols" _ _ _ _ _ _ _ _ _ bounds
    (by simpa only [Nat.mul_one] using hb.history) (by simp [historyPointer]) (by simp [vector]) (by simp [hc, vector]))
  intro h
  apply Stmt.TraceSafeListR.of_forall
  simp [Stmt.TraceSafeR, Op.SafeAtR, Op.SafeAtR.eq_def]

theorem post_safe (p : Layout) (l : Launch) (r : Regions) (scale : ℝ)
    (bounds : RegionBounds) (hb : Safe.TailBounds p l r bounds) (s : BlockState)
    (hh : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hr : s.regs .nat [p.BT] "rows" = some (vector p.BT (l.chunk*p.BT)))
    (hc : s.regs .nat [p.BV] "cols" = some (vector p.BV (l.value*p.BV))) :
    Stmt.TraceSafeListR .triv bounds (Program.tail p r scale ++ [Program.store p]) s := by
  unfold Program.tail Real.tail chunkGlaFront
  simp only [List.drop_succ_cons, List.drop_zero, List.take_succ_cons,
    List.take_zero, List.cons_append, List.nil_append]
  rw [cons_of_run (stepStmt_assign_eq_some (gatePtr_eval p l r s hh hr))]
  refine ⟨by simp [Stmt.TraceSafeR, Program.gatePtr, Program.times, Op.SafeAtR, Op.SafeAtR.eq_def], ?_⟩
  apply assign_cons (gate_safe p l r _ bounds hb.gate (by simp [vector]) (by simp [hr, vector]))
  intro gate
  apply assign_cons (by simp [Op.SafeAtR, Op.SafeAtR.eq_def])
  intro decayedO
  apply assign_cons (by simp [Op.SafeAtR, Op.SafeAtR.eq_def])
  intro decayedS
  apply assign_cons (by simp [Op.SafeAtR, Op.SafeAtR.eq_def])
  intro masked
  rw [cons_of_run (stepStmt_assign_eq_some
    (valuePtr_eval p l r.value _ (by simp [hh]) (by simp [hr, vector]) (by simp [hc, vector])))]
  refine ⟨by simp [Stmt.TraceSafeR, Program.valuePtr, Program.ptr2, Program.times,
    Program.scaleCol, Program.col, Program.row, Op.SafeAtR, Op.SafeAtR.eq_def], ?_⟩
  apply assign_cons (load2_safe _ r.value "p_v" "rows" "cols" _ _ _ _ _ _ _ _ _ bounds
    (by simpa only [Nat.mul_one] using hb.value) (by simp [valuePointer])
    (by simp [hr, vector]) (by simp [hc, vector]))
  intro value
  apply assign_cons (by simp [Op.SafeAtR, Op.SafeAtR.eq_def])
  intro output
  rw [cons_of_run (stepStmt_assign_eq_some
    (valuePtr_eval p l r.output _ (by simp [hh]) (by simp [hr, vector]) (by simp [hc, vector])))]
  refine ⟨by simp [Stmt.TraceSafeR, Program.valuePtr, Program.ptr2, Program.times,
    Program.scaleCol, Program.col, Program.row, Op.SafeAtR, Op.SafeAtR.eq_def], ?_⟩
  apply Stmt.TraceSafeListR.cons_intro
    (store_safe p l r _ bounds hb (by simp) (by simp [hr, vector]) (by simp [hc, vector]))
  intro _ _
  exact Stmt.TraceSafeListR.nil_intro

theorem loop_safe (p : Layout) (r : Regions) (s0 : BlockState) (bounds : RegionBounds)
    (hb : ∀ kb, kb < numBlocks p → Safe.StepBounds p (launch s0) r kb bounds)
    (u : BlockState) (hi : Execution.Holds p r s0 0 u) :
    (Program.loop p r).TraceSafeR .triv bounds u := by
  rw [Program.loop, Stmt.TraceSafeR]
  simp only [tripCount, Op.SafeAtR, evalOpR_triv, evalOp,
    Option.bind, Tile.scalar, Tile.bop_data, NumericDType.add, NumericDType.sub,
    NumericDType.div, Broadcast.leftIndex, Broadcast.rightIndex, true_and]
  apply Stmt.forRangeTraceSafeR_inv .triv bounds "i_k" (numBlocks p) 1
    (Program.iteration p r) (Execution.Holds p r s0)
  · intro kb s hn hs
    refine ⟨iteration_safe p (launch s0) r kb bounds (hb kb hn) _ ?_ ?_ ?_ ?_ (by simp), ?_⟩
    · simpa [launch] using hs.headReg
    · simpa [launch] using hs.timeReg
    · simpa [launch] using hs.rowsReg
    · simpa [launch] using hs.colsReg
    · simpa only [stepStmtsR_triv] using Execution.loop_step p r s0 kb s hn hs
  · exact hi

theorem source_safe (p : Layout) (r : Regions) (scale : ℝ) (s : BlockState)
    (bounds : RegionBounds) (hb : Safe.Bounds p (launch s) r bounds) :
    (Program.typed p r scale).eraseDType.toAlgKernel.TraceSafeR .triv bounds s := by
  rw [Kernel.TraceSafeR, Program.source_body, Program.body]
  simp only [List.append_assoc, List.singleton_append, List.cons_append]
  apply Stmt.TraceSafeListR.append_intro _ s
  · apply Stmt.TraceSafeListR.of_forall
    simp [Program.setup, Real.setup, Program.addresses, Program.indices, Program.times,
      Stmt.TraceSafeR, Op.SafeAtR, Op.SafeAtR.eq_def]
  · intro u he
    rw [stepStmtsR_triv, Execution.setup_executes] at he
    cases he
    obtain ⟨v, hl, hv⟩ := Execution.loop_executes p r s (Execution.start p s)
      (Execution.start_holds p r s)
    rw [cons_of_run hl]
    refine ⟨loop_safe p r s bounds hb.loop _ (Execution.start_holds p r s), ?_⟩
    exact post_safe p (launch s) r scale bounds hb.tail v hv.headReg hv.rowsReg hv.colsReg

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.IOSourceSafety
