import bench.tritonbench_g.pow_scalar_tensor.Proofs.Source
import bench.tritonbench_g.pow_scalar_tensor.Proofs.LegacyReal
import VeriTile.Triton.Semantics.BlockPadding

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.RealTransport
open VeriTile.Triton
set_option maxHeartbeats 400000
set_option maxRecDepth 16000

/-- Step equality under the explicit zero-undef condition, accompanied by
actual access-safety equivalence. A final-state equality alone is insufficient. -/
def Paired (lhs rhs : Stmt) : Prop :=
  (∀ s : BlockState, s.undef = (fun _ _ => 0) → stepStmt lhs s = stepStmt rhs s) ∧
  (∀ bounds s, s.undef = (fun _ _ => 0) →
    (Stmt.TraceSafe bounds lhs s ↔ Stmt.TraceSafe bounds rhs s))

theorem lists {lhs rhs : List Stmt} (hpair : List.Forall₂ Paired lhs rhs)
    (hflat : StmtList.FlattenOk rhs) (s : BlockState)
    (hu : s.undef = fun _ _ => 0) :
    stepStmts lhs s = stepStmts rhs s ∧
      ∀ bounds, Stmt.TraceSafeList bounds lhs s ↔ Stmt.TraceSafeList bounds rhs s := by
  induction hpair generalizing s with
  | nil => exact ⟨rfl, fun _ => Iff.rfl⟩
  | @cons a b left right hab hrest ih =>
    have hfl : b.FlattenOk ∧ StmtList.FlattenOk right := hflat
    have hs := hab.1 s hu
    cases he : stepStmt b s with
    | none =>
      constructor
      · simp only [stepStmts, hs, he, Option.bind_none]
      · intro bounds
        simp only [Stmt.TraceSafeList, hs, he, hab.2 bounds s hu]
    | some t =>
      have ht : t.undef = fun _ _ => 0 :=
        (stepStmt_undef b s t hfl.1 he).trans hu
      obtain ⟨hsteps, hsafe⟩ := ih hfl.2 t ht
      constructor
      · simpa only [stepStmts, hs, he, Option.bind_some] using hsteps
      · intro bounds
        simpa only [Stmt.TraceSafeList, hs, he, hab.2 bounds s hu,
          Option.some.injEq, forall_eq] using
            and_congr_right (fun _ => hsafe bounds)

theorem offset_eval (candidate : Bool) (B : Nat) (s : BlockState) :
    evalOp (RealSource.offset candidate B) s =
      evalOp (.mul .nat .nil (.ref .nat [] "tile_id0") (.constNat B)) s := by
  cases candidate
  · rfl
  · simp only [RealSource.offset, ↓reduceIte, evalOp_mul, evalOp_ref, evalOp_constNat]
    cases h : s.regs .nat [] "tile_id0" with
    | none => rfl
    | some v =>
      apply congrArg some
      ext i
      simp [Tile.bop, NumericDType.mul, Nat.mul_comm]

theorem cast_real (op : Op .real shape) (s : BlockState) :
    evalOp (.castFloat .real .real op) s = evalOp op s := by
  erw [evalOp_castFloat .real .real op s]
  cases h : evalOp op s <;> simp [h, FloatDType.cast]

private theorem flat_cast (v : ℝ) :
    (Op.castFloat .real .real (Op.const v)).FlattenOk := by
  rw [Op.FlattenOk]
  simp [Op.FlattenOk]

private theorem safe_cast (bounds : RegionBounds) (s : BlockState) (v : ℝ) :
    Op.SafeAt bounds s (Op.castFloat .real .real (Op.const v)) := by
  rw [Op.SafeAt]
  simp [Op.SafeAt]

/-- The new source and the historical per-tile mathematics agree after
explicitly choosing zero unspecified values. The actual source still retains
its omitted padding and fp32 conversion before erasure. -/
theorem tile_pairs (candidate : Bool) (base : ℝ) (X Y : RegionName)
    (sx sy N B : Nat) :
    List.Forall₂ Paired (RealSource.perTile candidate base X Y sx sy N B)
      (perTileStmts base X Y sx sy N B) := by
  unfold RealSource.perTile perTileStmts
  refine .cons ⟨fun _ _ => rfl, fun _ _ _ => Iff.rfl⟩
    (.cons ?_ (.cons ⟨fun _ _ => rfl, fun _ _ _ => Iff.rfl⟩
      (.cons ?_ (.cons ?_ (.cons ⟨fun _ _ => rfl, fun _ _ _ => Iff.rfl⟩
        (.cons ⟨fun _ _ => rfl, fun _ _ _ => Iff.rfl⟩ .nil))))))
  · constructor
    · intro s _
      simp only [stepStmt, offset_eval]
    · intro bounds s _
      cases candidate <;> simp [Stmt.TraceSafe, RealSource.offset, Op.SafeAt]
  · constructor
    · intro s hu
      simp only [stepStmt, evalOp_load_padding_of_undef_zero _ _ _ _ _ hu]
    · intro bounds s _
      simp only [Stmt.TraceSafe, Op.SafeAt]
  · constructor
    · intro s _
      have hc := cast_real (Op.const base) s
      simp only [stepStmt, evalOp_pow]
      erw [hc]
    · intro bounds s _
      simp [Stmt.TraceSafe, Op.SafeAt]
      exact safe_cast bounds s base

theorem tile_flat (base : ℝ) (X Y : RegionName) (sx sy N B : Nat) :
    StmtList.FlattenOk (perTileStmts base X Y sx sy N B) := by
  simp [perTileStmts, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  exact flat_cast base

theorem tile (candidate : Bool) (base : ℝ) (X Y : RegionName)
    (sx sy N B : Nat) (s : BlockState) (hu : s.undef = fun _ _ => 0) :
    stepStmts (RealSource.perTile candidate base X Y sx sy N B) s =
      stepStmts (perTileStmts base X Y sx sy N B) s ∧
    ∀ bounds, Stmt.TraceSafeList bounds (RealSource.perTile candidate base X Y sx sy N B) s ↔
      Stmt.TraceSafeList bounds (perTileStmts base X Y sx sy N B) s :=
  lists (tile_pairs candidate base X Y sx sy N B) (tile_flat base X Y sx sy N B) s hu

theorem same (st : Stmt) : Paired st st :=
  ⟨fun _ _ => rfl, fun _ _ _ => Iff.rfl⟩

theorem range_steps (index : RegName) (cur stop stride : Nat)
    (lhs rhs : List Stmt) (hpair : List.Forall₂ Paired lhs rhs)
    (hflat : StmtList.FlattenOk rhs) (s : BlockState)
    (hu : s.undef = fun _ _ => 0) :
    stepForRangeAux index cur stop stride lhs s =
      stepForRangeAux index cur stop stride rhs s := by
  by_cases hz : stride = 0
  · subst stride
    simp only [stepForRangeAux.step_zero_step]
  · have hs : 0 < stride := Nat.pos_of_ne_zero hz
    by_cases hc : cur < stop
    · have hb := (lists hpair hflat (s.setReg index .nat [] (Tile.scalar cur)) hu).1
      rw [stepForRangeAux.step_lt hz hc, stepForRangeAux.step_lt hz hc, hb]
      cases he : stepStmts rhs (s.setReg index .nat [] (Tile.scalar cur)) with
      | none => rfl
      | some t =>
          exact range_steps index (cur + stride) stop stride lhs rhs hpair hflat t
            ((stepStmts_undef rhs _ t hflat he).trans hu)
    · rw [stepForRangeAux.step_ge hz (by omega), stepForRangeAux.step_ge hz (by omega)]
termination_by stop - cur
decreasing_by omega

theorem range_safe (bounds : RegionBounds) (index : RegName) (cur stop stride : Nat)
    (lhs rhs : List Stmt) (hpair : List.Forall₂ Paired lhs rhs)
    (hflat : StmtList.FlattenOk rhs) (s : BlockState)
    (hu : s.undef = fun _ _ => 0) :
    Stmt.forRangeTraceSafe bounds index cur stop stride lhs s ↔
      Stmt.forRangeTraceSafe bounds index cur stop stride rhs s := by
  rw [Stmt.forRangeTraceSafe, Stmt.forRangeTraceSafe]
  by_cases hz : stride = 0
  · simp only [hz, ↓reduceIte]
  · simp only [hz, ↓reduceIte]
    by_cases hc : cur < stop
    · obtain ⟨heq, hsafe⟩ := lists hpair hflat (s.setReg index .nat [] (Tile.scalar cur)) hu
      simp only [hc, ↓reduceIte, hsafe bounds, heq]
      cases he : stepStmts rhs (s.setReg index .nat [] (Tile.scalar cur)) with
      | none => rfl
      | some t =>
          exact and_congr_right (fun _ => range_safe bounds index (cur + stride)
            stop stride lhs rhs hpair hflat t
            ((stepStmts_undef rhs _ t hflat he).trans hu))
    · simp only [hc, ↓reduceIte]
termination_by stop - cur
decreasing_by omega

theorem dynamic_range (index : RegName) (start stop stride : Op .nat [])
    (lhs rhs : List Stmt) (hpair : List.Forall₂ Paired lhs rhs)
    (hflat : StmtList.FlattenOk rhs) :
    Paired (.forRangeDyn index start stop stride lhs)
      (.forRangeDyn index start stop stride rhs) := by
  constructor
  · intro s hu
    simp only [stepForRangeAux.forRangeDyn_unfold]
    congr 1
    funext a
    congr 1
    funext b
    congr 1
    funext c
    exact range_steps _ _ _ _ _ _ hpair hflat s hu
  · intro bounds s hu
    simp only [Stmt.TraceSafe]
    cases evalOp start s <;> cases evalOp stop s <;> cases evalOp stride s <;>
      simp only [range_safe bounds index _ _ _ lhs rhs hpair hflat s hu]

def legacy (oneTile : Bool) (base : ℝ) (X Y : RegionName)
    (sx sy N tasks tiles B : Nat) : Kernel :=
  if oneTile then
    (pow_func_scalar_tensor_kernel_rank_1_one_tile_surface base X Y sx sy N tasks tiles B).toAlgKernel
  else
    (pow_func_scalar_tensor_kernel_rank_1_grid_stride_surface base X Y sx sy N tasks tiles B).toAlgKernel

theorem legacy_flat (oneTile : Bool) (base : ℝ) (X Y : RegionName)
    (sx sy N tasks tiles B : Nat) :
    (legacy oneTile base X Y sx sy N tasks tiles B).FlattenOk := by
  cases oneTile <;> unfold legacy <;> simp only [Bool.false_eq_true, ↓reduceIte]
  all_goals
    unfold Kernel.FlattenOk
    first | rw [oneTile_body_eq] | rw [gridStride_body_eq]
    simp [numTiles0Stmt, perTileStmts, StmtList.FlattenOk, Stmt.FlattenOk,
      Op.FlattenOk]
    exact flat_cast base

/-- Pair the actual erased source with the historical mathematical execution,
including all reached loop iterations; padding is reconciled only under the
explicit zero-undef premise used by the finite-memory bridge. -/
theorem whole_pairs (oneTile candidate : Bool) (base : ℝ) (X Y : RegionName)
    (sx sy N tasks tiles B : Nat) :
    List.Forall₂ Paired (RealSource.kernel oneTile candidate base X Y sx sy N tasks tiles B).body
      (legacy oneTile base X Y sx sy N tasks tiles B).body := by
  rw [RealSource.body_eq]
  cases oneTile
  · change List.Forall₂ Paired
      (_ :: _ :: _ :: _ :: []) (_ :: _ :: _ :: _ :: [])
    refine .cons (same _) (.cons (same _) (.cons (same _) (.cons ?_ .nil)))
    apply dynamic_range
    · exact .cons (same _) (tile_pairs candidate base X Y sx sy N B)
    · exact ⟨by simp [Stmt.FlattenOk, Op.FlattenOk, flat_cast], tile_flat base X Y sx sy N B⟩
  · change List.Forall₂ Paired (_ :: _ :: _ :: _) (_ :: _ :: _ :: _)
    exact .cons (same _) (.cons (same _) (.cons (same _)
      (tile_pairs candidate base X Y sx sy N B)))

theorem whole (oneTile candidate : Bool) (base : ℝ) (X Y : RegionName)
    (sx sy N tasks tiles B : Nat) (s : BlockState) (hu : s.undef = fun _ _ => 0) :
    exec (RealSource.kernel oneTile candidate base X Y sx sy N tasks tiles B) s =
      exec (legacy oneTile base X Y sx sy N tasks tiles B) s ∧
    ∀ bounds, (RealSource.kernel oneTile candidate base X Y sx sy N tasks tiles B).TraceSafe bounds s ↔
      (legacy oneTile base X Y sx sy N tasks tiles B).TraceSafe bounds s :=
  lists (whole_pairs oneTile candidate base X Y sx sy N tasks tiles B)
    (legacy_flat oneTile base X Y sx sy N tasks tiles B) s hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.RealTransport
