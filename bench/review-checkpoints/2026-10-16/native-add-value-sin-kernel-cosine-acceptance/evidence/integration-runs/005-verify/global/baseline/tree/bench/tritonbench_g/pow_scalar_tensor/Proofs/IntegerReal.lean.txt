import bench.tritonbench_g.pow_scalar_tensor.Proofs.IntegerGrid

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerReal
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

private theorem numTiles_step (N B : Nat) (s : BlockState) :
    stepStmt (IntegerSource.numTiles N B) s =
      some (s.setReg "num_tiles0" .nat [] (numTiles0Val N B)) := by
  exact stepStmt_assign_eq_some (by
    simp only [evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat,
      Option.bind_some, numTiles0Val]
    rfl)

private theorem singleton (st : Stmt) (s : BlockState) :
    stepStmts [st] s = stepStmt st s := by
  simp only [stepStmts]
  cases stepStmt st s <;> rfl

/-- Actual regional integer-source execution, safety and full memory frame.
The expected function is supplied independently of execution. -/
def Regional {α : Type} (k : Kernel) (bounds : RegionBounds) (s : BlockState)
    (Y : RegionName) (write : α → Nat) (active : α → Prop) (expected : α → ℝ) : Prop :=
  k.TraceSafe bounds s ∧
  ∃ t, exec k s = some t ∧
    (∀ i, active i → t.readMem Y (write i) = expected i) ∧
    (∀ r o, (r ≠ Y ∨ ∀ i, active i → o ≠ write i) → t.mem r o = s.mem r o)

/-- The complete single-tile branch supports arbitrary N, B, scalar base and
independent strides. Signed32 obligations concern exactly active input values;
the actual unspecified inactive values are integer zero under `hu`. -/
theorem one_tile (candidate : Bool) (base : ℝ) (X Y : RegionName)
    (sx sy N tasks tiles B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hstride : 0 < sy) (hu : s.undef = fun _ _ => 0)
    (hvalid : ∀ i : Fin B, taskIndex (s.pids 0) B i < N →
      Int32Bits.inRange (s.readMemValue .int X (taskIndex (s.pids 0) B i * sx)) = Bool.true)
    (hin : ∀ i : Fin B, taskIndex (s.pids 0) B i < N →
      taskIndex (s.pids 0) B i * sx < bounds X)
    (hout : ∀ i : Fin B, taskIndex (s.pids 0) B i < N →
      taskIndex (s.pids 0) B i * sy < bounds Y) :
    Regional (IntegerSource.kernel Bool.true candidate base X Y sx sy N tasks tiles B)
      bounds s Y (fun i : Fin B => taskIndex (s.pids 0) B i * sy)
      (fun i => taskIndex (s.pids 0) B i < N)
      (fun i => IntegerSteps.powSpec s X sx base (taskIndex (s.pids 0) B i)) := by
  let sp := ((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
    "num_tiles0" .nat [] (numTiles0Val N B)).setReg
    "tile_id" .nat [] (Tile.scalar (s.pids 0))
  obtain ⟨t,he,hv,_,_,_,hf⟩ := IntegerSteps.perTile_steps candidate base X Y sx sy N B
    hstride (s.pids 0) sp (Tile.scalar (s.pids 0)) (by simp [sp]) rfl hu hvalid
  refine ⟨?_,t,?_,hv,hf⟩
  · change Stmt.TraceSafeList bounds (IntegerSource.body Bool.true candidate base X Y sx sy N tiles B) s
    simp only [IntegerSource.body, ↓reduceIte, List.cons_append, List.nil_append]
    refine Stmt.TraceSafeList.cons_intro
      (by simp only [Stmt.TraceSafe, Op.SafeAt]) (fun t1 ht1 => ?_)
    rw [stepStmt_assign_eq_some (evalOp_programId 0 s)] at ht1
    obtain rfl := Option.some.inj ht1
    refine Stmt.TraceSafeList.cons_intro
      (by simp [IntegerSource.numTiles, Stmt.TraceSafe, Op.SafeAt]) (fun t2 ht2 => ?_)
    rw [numTiles_step] at ht2
    obtain rfl := Option.some.inj ht2
    refine Stmt.TraceSafeList.cons_intro
      (by simp only [Stmt.TraceSafe, Op.SafeAt]) (fun t3 ht3 => ?_)
    rw [stepStmt_assign_eq_some (show evalOp (.ref .nat [] "pid") _ =
      some (Tile.scalar (s.pids 0)) by simp)] at ht3
    obtain rfl := Option.some.inj ht3
    exact IntegerSteps.perTile_traceSafe candidate base X Y sx sy N B bounds (s.pids 0)
      sp (Tile.scalar (s.pids 0)) (by simp [sp]) rfl hin hout
  · change stepStmts (IntegerSource.body Bool.true candidate base X Y sx sy N tiles B) s = some t
    simp only [IntegerSource.body, ↓reduceIte, List.cons_append, List.nil_append]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s)),
      stepStmts.cons_some (numTiles_step N B _),
      stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.ref .nat [] "pid") _ =
        some (Tile.scalar (s.pids 0)) by simp))]
    exact he

/-- The full grid-stride branch, including its actual prologue and all loop
iterations, implements integer power of the initial typed input memory. -/
theorem grid (candidate : Bool) (base : ℝ) (X Y : RegionName)
    (sx sy N tasks tiles B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hstride : 0 < sy) (hdisj : X ≠ Y) (hgrid : 0 < s.numPids 0)
    (hu : s.undef = fun _ _ => 0)
    (hvalid : ∀ (j : Fin tiles) (i : Fin B), taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      Int32Bits.inRange (s.readMemValue .int X (taskIndex (s.pids 0+j.val*s.numPids 0) B i * sx)) = Bool.true)
    (hin : ∀ (j : Fin tiles) (i : Fin B), taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      taskIndex (s.pids 0+j.val*s.numPids 0) B i * sx < bounds X)
    (hout : ∀ (j : Fin tiles) (i : Fin B), taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      taskIndex (s.pids 0+j.val*s.numPids 0) B i * sy < bounds Y) :
    Regional (IntegerSource.kernel Bool.false candidate base X Y sx sy N tasks tiles B)
      bounds s Y (fun p : Fin tiles × Fin B => taskIndex (s.pids 0+p.1.val*s.numPids 0) B p.2 * sy)
      (fun p => taskIndex (s.pids 0+p.1.val*s.numPids 0) B p.2 < N)
      (fun p => IntegerSteps.powSpec s X sx base (taskIndex (s.pids 0+p.1.val*s.numPids 0) B p.2)) := by
  let sp := ((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
    "num_tiles0" .nat [] (numTiles0Val N B)).setReg
    "num_ctas" .nat [] (Tile.scalar (s.numPids 0))
  obtain ⟨hs,t,he,hv,hf⟩ := IntegerGrid.range_run candidate base X Y sx sy N B hstride
    hdisj (s.pids 0) (s.numPids 0) hgrid tiles bounds hin hout tiles 0 (by omega)
    sp (by simp [sp]) (by simp [sp]) hu (fun j i _ hj hi => hvalid ⟨j,hj⟩ i hi)
  refine ⟨?_,t,?_,fun p hp => hv p.1.val p.2 (Nat.zero_le _) p.1.isLt hp,?_⟩
  · change Stmt.TraceSafeList bounds (IntegerSource.body Bool.false candidate base X Y sx sy N tiles B) s
    simp only [IntegerSource.body, Bool.false_eq_true, ↓reduceIte, List.cons_append, List.nil_append]
    refine Stmt.TraceSafeList.cons_intro
      (by simp only [Stmt.TraceSafe, Op.SafeAt]) (fun t1 ht1 => ?_)
    rw [stepStmt_assign_eq_some (evalOp_programId 0 s)] at ht1
    obtain rfl := Option.some.inj ht1
    refine Stmt.TraceSafeList.cons_intro
      (by simp [IntegerSource.numTiles, Stmt.TraceSafe, Op.SafeAt]) (fun t2 ht2 => ?_)
    rw [numTiles_step] at ht2
    obtain rfl := Option.some.inj ht2
    refine Stmt.TraceSafeList.cons_intro
      (by simp only [Stmt.TraceSafe, Op.SafeAt]) (fun t3 ht3 => ?_)
    rw [stepStmt_assign_eq_some (evalOp_numPrograms 0 _)] at ht3
    obtain rfl := Option.some.inj ht3
    refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
    simpa only [Stmt.TraceSafe, Op.SafeAt, evalOp_constNat, Tile.scalar_data, true_and] using hs
  · change stepStmts (IntegerSource.body Bool.false candidate base X Y sx sy N tiles B) s = some t
    simp only [IntegerSource.body, Bool.false_eq_true, ↓reduceIte, List.cons_append, List.nil_append]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s)),
      stepStmts.cons_some (numTiles_step N B _),
      stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_numPrograms 0 _)),
      singleton, stepForRangeAux.forRangeDyn_unfold]
    simpa only [evalOp_constNat, Option.bind_some] using he
  · intro r o hc
    exact hf r o (hc.elim Or.inl (fun hh => Or.inr (fun j i _ hj hi => hh (⟨j,hj⟩,i) hi)))

theorem flattenOk (oneTile candidate : Bool) (base : ℝ) (X Y : RegionName)
    (sx sy N tasks tiles B : Nat) :
    (IntegerSource.kernel oneTile candidate base X Y sx sy N tasks tiles B).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [IntegerSource.body_eq]
  cases oneTile <;> cases candidate <;>
    simp [IntegerSource.body, IntegerSource.perTile, IntegerSource.loop, IntegerSource.numTiles,
      IntegerSource.offset, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerReal
