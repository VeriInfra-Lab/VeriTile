import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealLoop
import VeriTile.Triton.Memory.TraceProofs

/-! Active address safety of every value-block iteration. The four initial-array
windows use the exact source masks and strides; the existing full-loop
invariant carries the launch registers through all iterations. -/
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.RealLoopSafety
open VeriTile.Triton RealMath RealStep
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Bounds for initial-array windows. The history mask deliberately remains
NT*K, including source accesses beyond the logical last key-block lane. -/
structure Bounds (p : Layout) (l : Launch) (r : Regions) (bounds : RegionBounds) : Prop where
  value : ∀ (i : Fin p.BT) (j : Nat), l.time * p.BT + i.val < p.T → j < p.V →
    l.head * p.s_v_h + (l.time * p.BT + i.val) * p.s_v_t + j < bounds r.v
  dout : ∀ (i : Fin p.BT) (j : Nat), l.time * p.BT + i.val < p.T → j < p.V →
    l.head * p.s_v_h + (l.time * p.BT + i.val) * p.s_v_t + j < bounds r.dout
  history : ∀ (j : Nat) (e : Fin p.BK), j < p.V →
    l.key * p.BK + e.val < p.K →
    l.head * p.s_h_h + l.time * (p.K * p.s_h_t) + j + (l.key * p.BK + e.val) * p.s_h_t < bounds r.h
  dhist : ∀ (j : Nat) (e : Fin p.BK), j < p.V →
    l.key * p.BK + e.val < p.K →
    l.head * p.s_h_h + l.time * (p.K * p.s_h_t) + j + (l.key * p.BK + e.val) * p.s_h_t < bounds r.dh

theorem block_load_safe {shape : TileShape} (bounds : RegionBounds) (s : BlockState)
    (name : RegName) (ptrs : Tile .blockPtr shape) (bc : List Nat)
    (hp : s.regs .blockPtr shape name = some ptrs)
    (hb : ∀ i, (ptrs.data i).inBounds (TileShape.indexToList shape i) bc = Bool.true →
      (ptrs.data i).address (TileShape.indexToList shape i) < bounds (ptrs.data i).region) :
    (Op.load .real (.blockPtr (.ref .blockPtr shape name) bc) .none .zero).SafeAt bounds s := by
  rw [Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt], trivial, ?_⟩
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ps he i _
  rw [evalOp_ref, hp] at he
  cases he
  exact hb i

theorem value_safe (bounds : RegionBounds) (s : BlockState) (p : Layout) (l : Launch)
    (r : RegionName) (name : RegName) (c : Nat)
    (hp : s.regs .blockPtr [p.BT, p.BV] name = some (vPointer p l r c))
    (hb : ∀ (i : Fin p.BT) (j : Nat), l.time * p.BT + i.val < p.T → j < p.V →
      l.head * p.s_v_h + (l.time * p.BT + i.val) * p.s_v_t + j < bounds r) :
    (Op.load .real (.blockPtr (.ref .blockPtr [p.BT, p.BV] name) [0,1]) .none .zero).SafeAt bounds s := by
  apply block_load_safe bounds s name _ _ hp
  rintro ⟨i,j,⟨⟩⟩ hi
  simp only [vPointer, TileShape.indexToList, BlockPtr.inBounds_2d_offsets,
    decide_eq_true_eq] at hi
  simpa only [vPointer, TileShape.indexToList, BlockPtr.address_2d_offsets,
    Nat.mul_one, Nat.add_assoc] using hb i (c * p.BV + j.val) hi.1 hi.2

theorem history_safe (bounds : RegionBounds) (s : BlockState) (p : Layout) (l : Launch)
    (r : RegionName) (name : RegName) (c : Nat)
    (hp : s.regs .blockPtr [p.BV, p.BK] name = some (hPointer p l r c))
    (hb : ∀ (j : Nat) (e : Fin p.BK), j < p.V →
      l.key * p.BK + e.val < p.K →
      l.head * p.s_h_h + l.time * (p.K * p.s_h_t) + j + (l.key * p.BK + e.val) * p.s_h_t < bounds r) :
    (Op.load .real (.blockPtr (.ref .blockPtr [p.BV, p.BK] name) [0,1]) .none .zero).SafeAt bounds s := by
  apply block_load_safe bounds s name _ _ hp
  rintro ⟨j,e,⟨⟩⟩ hi
  simp only [hPointer, TileShape.indexToList, BlockPtr.inBounds_2d_offsets,
    decide_eq_true_eq] at hi
  simpa only [hPointer, TileShape.indexToList, BlockPtr.address_2d_offsets,
    Nat.mul_one, Nat.add_assoc] using hb (c * p.BV + j.val) e hi.1 hi.2

theorem iteration_safe (s : BlockState) (p : Layout) (l : Launch) (r : Regions)
    (bounds : RegionBounds) (hb : Bounds p l r bounds) (c : Nat)
    (hik : s.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hit : s.regs .nat [] "i_t" = some (Tile.scalar l.time))
    (hib : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hiv : s.regs .nat [] "i_v" = some (Tile.scalar c)) :
    Stmt.TraceSafeList bounds (code p r) s := by
  unfold code cbdLoopBody
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (make_pointer r.v p.T p.V p.s_v_t 1 p.BT p.BV _ _ _ s
      (l.head * p.s_v_h) (l.time * p.BT) (c * p.BV)
      (mul_ref s "i_bh" l.head p.s_v_h hib)
      (mul_ref s "i_t" l.time p.BT hit)
      (mul_ref s "i_v" c p.BV hiv)))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (make_pointer r.h p.V p.K 1 p.s_h_t p.BV p.BK _ _ _ _
      (l.head * p.s_h_h + l.time * (p.K * p.s_h_t)) (c * p.BV) (l.key * p.BK)
      (add_nat _ _ _ (l.head * p.s_h_h) (l.time * (p.K * p.s_h_t))
        (mul_ref _ "i_bh" l.head p.s_h_h (by simp [hib]))
        (mul_ref_product _ "i_t" l.time p.K p.s_h_t (by simp [hit])))
      (mul_ref _ "i_v" c p.BV (by simp [hiv]))
      (mul_ref _ "i_k" l.key p.BK (by simp [hik]))))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (make_pointer r.dout p.T p.V p.s_v_t 1 p.BT p.BV _ _ _ _
      (l.head * p.s_v_h) (l.time * p.BT) (c * p.BV)
      (mul_ref _ "i_bh" l.head p.s_v_h (by simp [hib]))
      (mul_ref _ "i_t" l.time p.BT (by simp [hit]))
      (mul_ref _ "i_v" c p.BV (by simp [hiv]))))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (make_pointer r.dh p.V p.K 1 p.s_h_t p.BV p.BK _ _ _ _
      (l.head * p.s_h_h + l.time * (p.K * p.s_h_t)) (c * p.BV) (l.key * p.BK)
      (add_nat _ _ _ (l.head * p.s_h_h) (l.time * (p.K * p.s_h_t))
        (mul_ref _ "i_bh" l.head p.s_h_h (by simp [hib]))
        (mul_ref_product _ "i_t" l.time p.K p.s_h_t (by simp [hit])))
      (mul_ref _ "i_v" c p.BV (by simp [hiv]))
      (mul_ref _ "i_k" l.key p.BK (by simp [hik]))))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  apply TraceProofs.assign_cons (value_safe bounds _ p l r.v "p_v" c (by simp [vPointer]) hb.value)
  intro vs
  apply TraceProofs.assign_cons (value_safe bounds _ p l r.dout "p_do" c (by simp [vPointer]) hb.dout)
  intro os
  apply TraceProofs.assign_cons (history_safe bounds _ p l r.h "p_h" c (by simp [hPointer]) hb.history)
  intro hs
  apply TraceProofs.assign_cons (history_safe bounds _ p l r.dh "p_dh" c (by simp [hPointer]) hb.dhist)
  intro ds
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe, Op.SafeAt]
  all_goals simp [Op.SafeAt.eq_def]


/-- Every reached iteration uses the same initial-array access windows.
The invariant is the one already proved for the complete value loop. -/
private theorem counted_safe {bounds : RegionBounds} {index : RegName} {stop : Nat}
    {body : List Stmt} {P : Nat → BlockState → Prop}
    (hsafe : ∀ i state, i<stop → P i state →
      Stmt.TraceSafeList bounds body (state.setReg index .nat [] (Tile.scalar i)))
    (hnext : ∀ i state next, i<stop → P i state →
      stepStmts body (state.setReg index .nat [] (Tile.scalar i))=some next → P (i+1) next)
    (cur : Nat) (state : BlockState) (ready : P cur state) :
    Stmt.forLoopTraceSafe bounds index cur stop body state := by
  rw [Stmt.forLoopTraceSafe]
  by_cases active : cur<stop
  · simp only [active,if_true]
    refine ⟨hsafe cur state active ready,?_⟩
    cases ran : stepStmts body (state.setReg index .nat [] (Tile.scalar cur)) with
    | none => trivial
    | some next => exact counted_safe hsafe hnext (cur+1) next (hnext cur state next active ready ran)
  · simp [active]
termination_by stop-cur
decreasing_by omega

theorem loop_safe (setup : BlockState) (p : Layout) (l : Launch) (r : Regions)
    (a : Inputs) (bounds : RegionBounds) (hb : Bounds p l r bounds)
    (hzero : setup.undef = fun _ _ => 0)
    (hmv : ∀ off, setup.readMem r.v off = a.v off)
    (hmh : ∀ off, setup.readMem r.h off = a.h off)
    (hmo : ∀ off, setup.readMem r.dout off = a.dout off)
    (hmd : ∀ off, setup.readMem r.dh off = a.dh off)
    (hik : setup.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hit : setup.regs .nat [] "i_t" = some (Tile.scalar l.time))
    (hib : setup.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hinit : RealLoop.Holds setup p l a 0 setup) :
    Stmt.TraceSafe bounds (RealLoop.statement p r) setup := by
  rw [RealLoop.statement, Stmt.TraceSafe]
  apply counted_safe (P := RealLoop.Holds setup p l a)
  · intro c s _ hi
    apply iteration_safe _ p l r bounds hb c
    · simpa using (hi.fixed .nat [] "i_k" (by decide)).trans hik
    · simpa using (hi.fixed .nat [] "i_t" (by decide)).trans hit
    · simpa using (hi.fixed .nat [] "i_bh" (by decide)).trans hib
    · simp
  · intro c s t hc hi he
    obtain ⟨u, hu, hf⟩ := RealLoop.iteration setup p l r a hzero hmv hmh hmo hmd hik hit hib c s hc hi
    rw [he] at hu
    cases hu
    exact hf
  · exact hinit

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.RealLoopSafety
