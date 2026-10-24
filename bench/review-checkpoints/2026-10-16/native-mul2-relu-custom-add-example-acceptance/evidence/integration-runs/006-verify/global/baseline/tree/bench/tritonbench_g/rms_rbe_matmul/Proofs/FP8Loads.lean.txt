import bench.tritonbench_g.rms_rbe_matmul.Proofs.Transport
import bench.tritonbench_g.attention_llama.Proofs.FP8RealLoads

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.ByteLoads
open VeriTile.Triton
open AttentionLlama.FP8RealLoads
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

private theorem overwrite (s : BlockState) (name : RegName)
    (a : Tile da sha) (b : Tile db shb) :
    (s.setReg name da sha a).setReg name db shb b = s.setReg name db shb b := by
  apply BlockState.ext <;> intros <;> try rfl
  simp only [BlockState.setReg]
  split <;> rfl

/-- Exact execution of all raw-load/bitcast/downcast assignments. Matrix weights
have the source's extra fp32-to-fp16 conversion; erasure retains its assignment. -/
theorem conversion_extra (name : RegName) (bytes : Op .int sh) (extra : Bool) (s : BlockState) :
    stepStmts (conversion name bytes ++
      (if extra then [.assign .real sh name (.ref .real sh name)] else [])) s =
      stepStmts [.assign .real sh name (.decodeFloat8E5 .real bytes)] s := by
  rw [ProgramRewrite.append, fuse_conversion]
  cases extra
  · simp [stepStmts]
  · simp only [↓reduceIte, stepStmts, stepStmt]
    cases he : evalOp (.decodeFloat8E5 .real bytes) s with
    | none => erw [he]; rfl
    | some tile => erw [he]; simp [evalOp_ref, overwrite]

/-- Fuse only a source weight conversion; decoding and every failure case
remain represented by the resulting expression. -/
def lowered : Stmt → Stmt
  | .assign .real sh name (.load .real (.ptr ptr) .none) =>
      if name == "rms_w" || name == "w" then
        .assign .real sh name (.decodeFloat8E5 .real (.load .int (.ptr ptr) .none))
      else .assign .real sh name (.load .real (.ptr ptr) .none)
  | st => st

theorem weight_load (name : RegName) (ptr : Op .ptr sh)
    (hn : name = "rms_w" ∨ name = "w") (s : BlockState) :
    stepStmts (RealSource.weightLoad Bool.true (.assign .real sh name (.load .real (.ptr ptr) .none))) s =
      stepStmts [lowered (.assign .real sh name (.load .real (.ptr ptr) .none))] s := by
  rcases hn with rfl | rfl
  · simpa [RealSource.weightLoad, lowered, conversion] using
      conversion_extra "rms_w" (.load .int (.ptr ptr) .none) Bool.false s
  · simpa [RealSource.weightLoad, lowered, conversion] using
      conversion_extra "w" (.load .int (.ptr ptr) .none) Bool.true s

/-- The fused reference iteration preserves native integer power and both
raw-byte loads/decoders. Erasure removes no decoder failure. -/
def loop (BM BN BK sxk swk srms : Nat) : List Stmt :=
  (rrmLoopBody BM BN BK sxk swk srms).map lowered

private theorem map_execution (code : List Stmt)
    (h : ∀ st ∈ code, ∀ s, stepStmts (RealSource.weightLoad Bool.true st) s = stepStmts [lowered st] s)
    (s : BlockState) :
    stepStmts (code.flatMap (RealSource.weightLoad Bool.true)) s = stepStmts (code.map lowered) s := by
  induction code generalizing s with
  | nil => rfl
  | cons head rest ih =>
    simp only [List.flatMap_cons, List.map_cons, ProgramRewrite.append, h head (by simp)]
    have ht := ih (fun st hs => h st (by simp [hs]))
    simp only [stepStmts]
    cases stepStmt (lowered head) s <;> simp [ht]

/-- Full iteration equality includes invalid bytes and NaN/Inf decode failure;
no finite-input hypothesis or floating-point admission law is used. -/
theorem source_iteration (BM BN BK sxk swk srms : Nat) (s : BlockState) :
    stepStmts (RealSource.loop Bool.true BM BN BK sxk swk srms) s =
      stepStmts (loop BM BN BK sxk swk srms) s := by
  let base := (rrmLoopBody BM BN BK sxk swk srms).map Stmt.eraseDType
  have hm : ∀ st ∈ base, ∀ s, stepStmts (RealSource.weightLoad Bool.true st) s = stepStmts [lowered st] s := by
    intro st hs t
    simp only [base, rrmLoopBody, List.map_cons, List.map_nil,
      Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      IntegralDType.eraseDType, ComparableDType.eraseDType,
      List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first | rfl | exact weight_load _ _ (by simp) t
  change stepStmts (base.flatMap (RealSource.weightLoad Bool.true)) s = _
  rw [map_execution base hm]
  simp only [base, loop, rrmLoopBody, List.map_cons, List.map_nil, lowered,
    Bool.false_eq_true, Bool.or_eq_true, beq_iff_eq, String.reduceEq,
    false_or, or_false, ↓reduceIte, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
    IntegralDType.eraseDType, ComparableDType.eraseDType, stepStmts, stepStmt,
    evalOp_add, evalOp_libdevicePowi, RealTransport.real_cast,
    eq_mpr_eq_cast, eq_mp_eq_cast, cast_eq]
  rfl

/-- A decoder result is demanded explicitly, with no default real number. -/
def FiniteAt (s : BlockState) (r : RegionName) (o : Nat) : Prop :=
  Float8E5Bits.isInt8 (s.readMemValue .int r o) = Bool.true ∧
    (Float8E5Bits.ofInt (s.readMemValue .int r o)).decodeRat.isSome = Bool.true

/-- The input view decodes finite weight cells and retains all other original
cells, including nonfinite or invalid bytes. Only finite loaded cells are used
by the real recurrence; no NaN/Inf encoding is assigned a real zero. -/
noncomputable def view (W RMS : RegionName) (s : BlockState) : BlockState :=
  {s with mem := fun r o =>
    if r = W ∨ r = RMS then
      if Float8E5Bits.isInt8 (s.readMemValue .int r o) then
        match (Float8E5Bits.ofInt (s.readMemValue .int r o)).decodeRat with
        | some q => MemCell.real (q : ℝ)
        | none => s.mem r o
      else s.mem r o
    else s.mem r o}

theorem view_outside (W RMS : RegionName) (s : BlockState) (r : RegionName) (o : Nat)
    (hW : r ≠ W) (hr : r ≠ RMS) :
    (view W RMS s).mem r o = s.mem r o := by simp [view, hW, hr]

/-- In this real projection a finite source byte has the same value as the
corresponding cell in the representation view. -/
theorem decoded_view (W RMS : RegionName) (s : BlockState) (r : RegionName) (o : Nat)
    (hw : r = W ∨ r = RMS) (h : FiniteAt s r o) :
    (Float8E5Bits.ofInt (s.readMemValue .int r o)).decodeRat.map (fun q : ℚ => (q : ℝ)) =
      some ((view W RMS s).readMem r o) := by
  rcases h with ⟨hb, hf⟩
  cases hd : (Float8E5Bits.ofInt (s.readMemValue .int r o)).decodeRat with
  | none => simp [hd] at hf
  | some q => simp [view, hw, hb, hd, BlockState.readMem, MemCell.real]

private theorem decode_eval (bytes : Op .int sh) (s : BlockState) :
    @evalOp .real sh (.decodeFloat8E5 .real bytes) s =
      (evalOp bytes s).bind (Float8E5Bits.decodeTile .real) := by
  rw [evalOp.eq_def]
  rfl

/-- A whole unmasked tile of finite raw weights decodes to the real input view. -/
theorem load (W RMS : RegionName) (s : BlockState) (name : RegName) (p : Tile .ptr sh)
    (hp : s.regs .ptr sh name = some p)
    (hw : ∀ i, (p.data i).1 = W ∨ (p.data i).1 = RMS)
    (hf : ∀ i, FiniteAt s (p.data i).1 (p.data i).2) :
    @evalOp .real sh (.decodeFloat8E5 .real (.load .int (.ptr (.ref .ptr sh name)) .none)) s =
      some (⟨fun i => some ((view W RMS s).readMem (p.data i).1 (p.data i).2)⟩ : Tile .real sh) := by
  let raw : Tile .int sh := ⟨fun i => s.readMemValue .int (p.data i).1 (p.data i).2⟩
  let vals : TileIndex sh → ℚ := fun i =>
    (Float8E5Bits.ofInt (raw.data i)).decodeRat.get (hf i).2
  have hd : Float8E5Bits.decodeTile .real raw = some (⟨fun i => some (vals i : ℝ)⟩ : Tile .real sh) := by
    apply Float8E5Bits.decodeTile_of_finite .real raw vals
    · exact fun i => (hf i).1
    · intro i
      simp [vals]
  rw [decode_eval]
  have hl : evalOp (.load .int (.ptr (.ref .ptr sh name)) .none) s = some raw := by
    simp only [evalOp, hp, bind, Option.bind_some, raw, ↓reduceIte]
  rw [hl]
  simp only [Option.bind_some, hd]
  congr 1
  ext i
  have hv := decoded_view W RMS s (p.data i).1 (p.data i).2 (hw i) (hf i)
  cases he : (Float8E5Bits.ofInt (s.readMemValue .int (p.data i).1 (p.data i).2)).decodeRat with
  | none => have := (hf i).2; simp [he] at this
  | some q =>
      have hval : (q : ℝ) = (view W RMS s).readMem (p.data i).1 (p.data i).2 := by simpa [he] using hv
      apply congrArg (fun q : ℝ => (some q : WithBot ℝ))
      simpa only [vals, raw, he, Option.get_some] using hval

@[simp] theorem view_setReg (W RMS : RegionName) (s : BlockState)
    (name : RegName) (v : Tile d sh) :
    view W RMS (s.setReg name d sh v) = (view W RMS s).setReg name d sh v := rfl

theorem view_memory (W RMS : RegionName) (s t : BlockState) (h : s.mem = t.mem) :
    (view W RMS s).mem = (view W RMS t).mem := by
  funext r o
  simp only [view, BlockState.readMemValue, BlockState.readMemTyped, h]

theorem finite_memory (s t : BlockState) (h : s.mem = t.mem) (r : RegionName) (o : Nat) :
    FiniteAt s r o ↔ FiniteAt t r o := by
  simp only [FiniteAt, BlockState.readMemValue, BlockState.readMemTyped, h]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.ByteLoads
