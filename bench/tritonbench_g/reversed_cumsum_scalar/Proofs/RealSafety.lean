import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealExecution
import VeriTile.Triton.Memory.TraceProofs

/-! Address safety of every load and store in the complete source loop. The
boundary checks exclude unspecified tail lanes from the memory footprint. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealSafety
open VeriTile.Triton RealProgram RealValues RealLoop
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

structure Bounds (T row : Nat) (s o : RegionName) (bounds : RegionBounds) : Prop where
  input : ∀ t, t < T → address T row t < bounds s
  output : ∀ t, t < T → address T row t < bounds o

private theorem pointer_bounds (T BT row : Nat) (r : RegionName) (c : Nat)
    (bounds : RegionBounds) (hb : ∀ t, t < T → address T row t < bounds r)
    (idx : TileIndex [BT])
    (hi : ((pointer T BT row r c).data idx).inBounds (TileShape.indexToList _ idx) [0] = Bool.true) :
    ((pointer T BT row r c).data idx).address (TileShape.indexToList _ idx) < bounds r := by
  obtain ⟨i,⟨⟩⟩ := idx
  simp only [pointer, TileShape.indexToList, BlockPtr.inBounds_1d, decide_eq_true_eq] at hi
  simpa only [pointer, TileShape.indexToList, BlockPtr.address_1d, Nat.mul_one, address] using
    hb (c * BT + i.val) hi

private theorem load_safe (T BT row : Nat) (r : RegionName) (c : Nat)
    (bounds : RegionBounds) (u : BlockState)
    (hp : u.regs .blockPtr [BT] "p_s" = some (pointer T BT row r c))
    (hb : ∀ t, t < T → address T row t < bounds r) :
    (Op.load .real (.blockPtr (.ref .blockPtr [BT] "p_s") [0]) .none .unspecified).SafeAt bounds u := by
  rw [Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt], trivial, ?_⟩
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ps he i _
  rw [evalOp_ref, hp] at he
  cases he
  exact pointer_bounds T BT row r c bounds hb i

private theorem store_safe (T BT row : Nat) (o : RegionName) (c : Nat)
    (bounds : RegionBounds) (u : BlockState)
    (hp : u.regs .blockPtr [BT] "p_o" = some (pointer T BT row o c))
    (hb : ∀ t, t < T → address T row t < bounds o) :
    Stmt.TraceSafe bounds (.store .real [BT]
      (.blockPtr (.ref .blockPtr [BT] "p_o") [0]) (.ref .real [BT] "b_o") .none) u := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt, Op.SafeAt], by simp [Op.SafeAt], trivial, ?_⟩
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ps he i _
  rw [evalOp_ref, hp] at he
  cases he
  exact pointer_bounds T BT row o c bounds hb i

theorem iteration_safe (swapped : Bool) (T BT row : Nat) (s o : RegionName)
    (bounds : RegionBounds) (hb : Bounds T row s o bounds) (j : Int) (u : BlockState)
    (hj : j < 1) (hr : u.regs .int [] "__rev_i_t" = some (Tile.scalar j))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar row)) :
    Stmt.TraceSafeList bounds (loopBody swapped T BT s o) u := by
  unfold loopBody
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (RealStep.index_eval j u hj hr))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (RealStep.pointer_eval swapped
    T BT row s (-j).toNat _ (by simp [hh]) (by simp)))]
  refine ⟨by cases swapped <;> simp [Stmt.TraceSafe, pointerOp, TypedProgram.offsetOp, Op.SafeAt], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (RealStep.pointer_eval Bool.false
    T BT row o (-j).toNat _ (by simp [hh]) (by simp)))]
  refine ⟨by simp [Stmt.TraceSafe, pointerOp, TypedProgram.offsetOp, Op.SafeAt], ?_⟩
  apply TraceProofs.assign_cons (load_safe T BT row s (-j).toNat bounds _ (by simp) hb.input)
  intro bs
  apply TraceProofs.assign_cons
  · simp [Op.SafeAt, Op.SafeAt.eq_def]
  intro bzz
  apply TraceProofs.assign_cons
  · simp [Op.SafeAt, Op.SafeAt.eq_def]
  intro bz
  apply TraceProofs.assign_cons
  · simp [outputOp, Op.SafeAt, Op.SafeAt.eq_def]
  intro bo
  apply Stmt.TraceSafeList.cons_intro (store_safe T BT row o (-j).toNat bounds _ (by simp) hb.output)
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

theorem loop_safe (initial : BlockState) (swapped : Bool) (T BT row : Nat)
    (s o : RegionName) (a : Nat → ℝ) (hSO : s ≠ o)
    (bounds : RegionBounds) (hb : Bounds T row s o bounds) (u : BlockState)
    (h : Holds initial T BT row s o a (numChunks T BT) u) :
    Stmt.TraceSafe bounds (statement swapped T BT s o) u := by
  rw [statement, Stmt.TraceSafe]
  refine ⟨by simp [TypedProgram.startOp, TypedProgram.countOp, Op.SafeAt],
    by simp [Op.SafeAt], by simp [Op.SafeAt], ?_⟩
  rw [start_eval]
  simp only [evalOp]
  change Stmt.forRangeIntTraceSafe bounds "__rev_i_t" (1 - (numChunks T BT : Int))
    1 1 (loopBody swapped T BT s o) u
  apply ReversedCumsum.SignedRange.safe bounds "__rev_i_t" 1 1 (loopBody swapped T BT s o)
    (fun j v => Holds initial T BT row s o a (SignedIndex.remaining j) v)
  · intro j v hj hv
    exact iteration_safe swapped T BT row s o bounds hb j _ hj (by simp) (by simpa using hv.head)
  · intro j v w hj hv he
    obtain ⟨w', he', hw⟩ := RealExecution.iteration initial swapped T BT row s o a hSO j v hj hv
    rw [he] at he'
    cases he'
    exact hw
  · simpa only [SignedIndex.at_start] using h

theorem source_safe (swapped : Bool) (T BT : Nat) (s o : RegionName)
    (dtype out_dtype : FloatDType) (u : BlockState) (hSO : s ≠ o) (hBT : 0 < BT)
    (bounds : RegionBounds) (hb : Bounds T (u.pids 0) s o bounds) :
    (TypedSource.source swapped dtype out_dtype s o T BT).eraseDType.toAlgKernel.TraceSafe bounds u := by
  rw [Kernel.TraceSafe, source_body, body]
  apply TraceProofs.append_safe
  · apply TraceProofs.all_safe
    simp [initCode, Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def]
  · intro v he
    rw [RealInit.executes] at he
    cases he
    apply Stmt.TraceSafeList.cons_intro
    · exact loop_safe u swapped T BT (u.pids 0) s o (u.readMem s) hSO bounds hb
        (RealInit.initial u) (RealInit.invariant T BT u s o (u.readMem s) (fun _ => rfl) hBT)
    · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem source_flatten_ok (swapped : Bool) (T BT : Nat) (s o : RegionName)
    (dtype out_dtype : FloatDType) :
    (TypedSource.source swapped dtype out_dtype s o T BT).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk, source_body]
  cases swapped <;> simp [body, statement, initCode, loopBody, outputOp, pointerOp,
    TypedProgram.offsetOp, TypedProgram.startOp, TypedProgram.countOp,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

#axiomsClean iteration_safe
#axiomsClean loop_safe
#axiomsClean source_safe
#axiomsClean source_flatten_ok
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealSafety
