import bench.tritonbench_g.kldiv_ops.Proofs.SourceDomain
import bench.tritonbench_g.kldiv_ops.Proofs.FP
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Meta.StatementAudit

/-! Nonempty typed executions through both blocks, including the partial final
block. This model realizes the used arithmetic exactly over reals; it is not an
IEEE implementation or a native compilation certificate. No numerical law is
required by the existing exact-address candidate. -/
namespace VeriTile.Bench.TritonBenchG.KldivOps.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
noncomputable section
open scoped Classical
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def model : Algebra (WithBot ℝ) where
  literal := fun _ _ r ↦ some r
  negInf := none
  binary := fun _ _ op a b ↦ match op with
    | .add => WithBot.realAdd a b
    | .sub => WithBot.realSub a b
    | .mul => WithBot.realMul a b
    | .div => WithBot.realDiv a b
    | .max => max a b
    | .pow => some 0
  unary := fun _ op a ↦ match op with
    | .log => WithBot.realLog a
    | .exp => WithBot.realExp a
    | .neg _ => WithBot.map Neg.neg a
    | _ => some 0
  compareLt := fun _ _ ↦ some (fun a b ↦ decide (a < b))
  compareLe := fun _ _ ↦ some (fun a b ↦ decide (a ≤ b))
  compareEq := fun _ _ ↦ some (fun a b ↦ decide (a = b))
  cast := fun _ _ _ x ↦ x
  fromNat := fun _ n ↦ some n
  fromInt := fun _ n ↦ some n
  fp32Bits := fun _ ↦ some 0
  fp32Load := id
  reduceMax := fun _ {_} ax keep v ↦
    ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ ↦ none⟩).data
  reduceSum := fun _ {_} ax keep v ↦ (Tile.reduceSum ax keep ⟨v⟩).data
  reduceSumTyped := fun _ d {_} ax keep v ↦
    some (fun i ↦ ofFloat d ((Tile.reduceSum ax keep ⟨fun j ↦ toFloat d (v j)⟩).data i))

/-- Independent strides, a nonzero program id, and a distinct tagged output sentinel. -/
def initial (flag : Bool) : State (WithBot ℝ) where
  mem := fun r _ ↦
    if r = "Y" then .mk .fp16 (some 2)
    else if r = "T" then .mk .real (some (if flag then 0 else 1))
    else .mk .bf16 (some 7)
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax = 0 then 1 else 0
  numPids := fun _ ↦ 2
  undef := fun d _ _ ↦ defaultValue d

def forwardKernel (v flag : Bool) (reduction : Fin 2) : ComputeKernel :=
  if flag then forwardLog .fp16 .fp32 .bf16 v "Y" "T" "O" 5 7 9 3 2 1 reduction.val
  else forwardDefault .fp16 .fp32 .bf16 v "Y" "T" "O" 5 7 9 3 2 1 reduction.val

def backwardKernel (v flag : Bool) : ComputeKernel :=
  if flag then backwardLog .fp32 .bf16 v "T" "O" 7 9 3 2
  else backwardDefault .fp32 .bf16 v "T" "O" 7 9 3 2

/-- The witness profiles satisfy the same frontend admission used publicly. -/
theorem forward_admitted (flag : Bool) (reduction : Fin 2) :
    ForwardSourceSupported .fp16 .fp32 .bf16 flag reduction.val := by
  constructor
  · intro _
    exact Or.inl rfl
  · exact Or.inr ⟨by decide, by decide⟩

theorem backward_admitted (flag : Bool) :
    BackwardSourceSupported .fp32 .bf16 flag := by
  cases flag
  · exact ⟨by decide, by decide⟩
  · exact Or.inl rfl

/-- Unroll the two source iterations before simplifying their execution. This
keeps the recursive interpreter from repeatedly expanding symbolic states. -/
private theorem range_three_two {α : Type} [Inhabited α] (M : Algebra α)
    (body : List ComputeStmt) (s : State α) :
    rangeInt M "i" 0 3 2 body s =
      (run M body (s.setReg "i" .int [] (fun _ ↦ 0))).bind (fun t ↦
        run M body (t.setReg "i" .int [] (fun _ ↦ 2))) := by
  simp [rangeInt]

/-- Actual typed execution for both target flags, reduction modes and candidates.
Three columns and block size two force both the loop and its masked tail to run. -/
def forward_original_run (flag : Bool) (reduction : Fin 2) :
    {t : State (WithBot ℝ) // run (engine model)
      (forwardKernel Bool.false flag reduction).surfaceBody (initial flag) = some t} := by
  refine Fin.cases ?_ (fun r ↦ Fin.cases ?_ (fun k ↦ Fin.elim0 k) r) reduction
  all_goals cases flag
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 1000000) [forwardKernel, forwardDefault, forwardDefaultOriginal,
      forwardDefault_fp16_fp32, forwardLog, forwardLogOriginal, forwardLog_fp16_fp32,
      commuteOutput, ComputeKernel.surfaceBody, run, step, range_three_two,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, bop, numericLt, numericLe, numericEq, natLt, initial, Cell.read,
      ofFloat, toFloat, TileShape.axisDim, Region.cast, FP.Structural.store, Tile.select, Tile.ptrAdd,
      TileShape.allIndices, List.finRange_succ]
    try rfl

def backward_original_run (flag : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (backwardKernel Bool.false flag).surfaceBody (initial flag) = some t} := by
  cases flag
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 1000000) [backwardKernel, backwardDefault, backwardDefaultOriginal,
      backwardLog, backwardLogOriginal, commuteOutput, ComputeKernel.surfaceBody,
      run, step, range_three_two, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      evalOffsets, evalDeltas, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, numericLt, numericLe, numericEq, natLt, initial,
      Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, FP.Structural.store, Tile.select, Tile.ptrAdd,
      TileShape.allIndices, List.finRange_succ]
    try rfl

/-- Candidate execution is transported by the full typed equivalence, sharing
its actual successful source state rather than re-expanding a second program. -/
def forward_run (v flag : Bool) (reduction : Fin 2) :
    {t : State (WithBot ℝ) // run (engine model)
      (forwardKernel v flag reduction).surfaceBody (initial flag) = some t} := by
  cases v
  · exact forward_original_run flag reduction
  · refine ⟨(forward_original_run flag reduction).val, ?_⟩
    have he := (forward_original_run flag reduction).property
    cases flag
    · exact (FPProof.forward_default_equivalence model .fp16 .fp32 .bf16
        "Y" "T" "O" 5 7 9 3 2 1 reduction.val (initial Bool.false)).symm.trans he
    · exact (FPProof.forward_log_equivalence model .fp16 .fp32 .bf16
        "Y" "T" "O" 5 7 9 3 2 1 reduction.val (initial Bool.true)).symm.trans he

def backward_run (v flag : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (backwardKernel v flag).surfaceBody (initial flag) = some t} := by
  cases v
  · exact backward_original_run flag
  · refine ⟨(backward_original_run flag).val, ?_⟩
    have he := (backward_original_run flag).property
    cases flag
    · exact (FPProof.backward_default_equivalence model .fp32 .bf16
        "T" "O" 7 9 3 2 (initial Bool.false)).symm.trans he
    · exact (FPProof.backward_log_equivalence model .fp32 .bf16
        "T" "O" 7 9 3 2 (initial Bool.true)).symm.trans he

private theorem forward_run_value (v flag : Bool) (reduction : Fin 2) :
    (forward_run v flag reduction).val = (forward_original_run flag reduction).val := by
  cases v <;> cases flag <;> rfl

private theorem backward_run_value (v flag : Bool) :
    (backward_run v flag).val = (backward_original_run flag).val := by
  cases v <;> cases flag <;> rfl

private theorem fin_two_cases_one {motive : Fin 2 → Type}
    (zero : motive 0) (succ : ∀ i : Fin 1, motive i.succ) :
    Fin.cases (motive := motive) zero succ (1 : Fin 2) = succ 0 := rfl

private theorem sum_fin2 {α : Type*} [AddCommMonoid α]
    (I : Fintype (Fin 2)) (f : Fin 2 → α) :
    @Finset.sum (Fin 2) α _ (@Finset.univ _ I) f = f 0 + f 1 := by
  have hI : I = Fin.fintype 2 := Subsingleton.elim _ _
  rw [hI]
  simp [Fin.sum_univ_succ]

/-- Both target branches produce nonzero losses. Reduction writes the accumulated
whole-row value once; elementwise mode writes all three active lanes. -/
theorem forward_cells (v flag : Bool) (reduction : Fin 2) :
    (forward_run v flag reduction).val.mem "O" 9 =
      Cell.mk .bf16 (some (if reduction.val = 0 then -2 else -6)) ∧
    (forward_run v flag reduction).val.mem "O" 10 =
      Cell.mk .bf16 (some (if reduction.val = 0 then -2 else 7)) ∧
    (forward_run v flag reduction).val.mem "O" 11 =
      Cell.mk .bf16 (some (if reduction.val = 0 then -2 else 7)) ∧
    (forward_run v flag reduction).val.mem "O" 8 = Cell.mk .bf16 (some 7) ∧
    (forward_run v flag reduction).val.mem "O" 12 = Cell.mk .bf16 (some 7) := by
  simp only [forward_run_value]
  cases flag <;> fin_cases reduction
  all_goals simp (maxSteps := 1000000) [forward_original_run, fin_two_cases_one, initial, model,
    FP.Structural.store, State.write, State.setReg, Cell.read, Region.cast,
    Tile.scalar, -Tile.scalar_eta, Function.comp_def, Tile.select, Tile.ptrAdd,
    TileShape.allIndices, List.finRange_succ, List.foldl, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim, TileShape.insertAxisIndex,
    TileShape.eraseAxis, sum_fin2, WithBot.realAdd, WithBot.realSub,
    WithBot.realMul, WithBot.realLog, WithBot.realExp]
  all_goals norm_num [WithBot.some_eq_coe, ← WithBot.coe_add, Option.map₂, Option.map, Option.bind,
    bop, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar, -Tile.scalar_eta,
    Function.comp_def, List.finRange_succ, List.foldl, List.flatMap, sum_fin2]
  all_goals first | rfl | (repeat' constructor)
  all_goals
    have h : ¬ (1 : WithBot ℝ) < 0 := by
      norm_num [← WithBot.coe_one, ← WithBot.coe_zero]
    simp only [if_neg h]
    norm_num [← WithBot.coe_one, ← WithBot.coe_zero, ← WithBot.coe_add,
      WithBot.some_eq_coe]
    rfl

/-- The full backward loop writes every active gradient and preserves the
masked fourth lane and the preceding sentinel, in both target branches. -/
theorem backward_cells (v flag : Bool) :
    (backward_run v flag).val.mem "O" 9 = Cell.mk .bf16 (some (-1)) ∧
    (backward_run v flag).val.mem "O" 10 = Cell.mk .bf16 (some (-1)) ∧
    (backward_run v flag).val.mem "O" 11 = Cell.mk .bf16 (some (-1)) ∧
    (backward_run v flag).val.mem "O" 8 = Cell.mk .bf16 (some 7) ∧
    (backward_run v flag).val.mem "O" 12 = Cell.mk .bf16 (some 7) := by
  simp only [backward_run_value]
  cases flag
  all_goals simp (maxSteps := 1000000) [backward_original_run, initial, model,
    FP.Structural.store, State.write, State.setReg, Cell.read, Region.cast,
    Tile.scalar, -Tile.scalar_eta, Function.comp_def, Tile.select, Tile.ptrAdd,
    TileShape.allIndices, List.finRange_succ, List.foldl, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realExp]
  all_goals norm_num [State.setReg, State.write, Cell.read, Region.cast,
    WithBot.some_eq_coe, Option.map₂, Option.map, Option.bind,
    bop, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar, -Tile.scalar_eta,
    Function.comp_def, List.finRange_succ, List.foldl, List.flatMap]
  all_goals first | rfl | (repeat' constructor)

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivOps.NumericExecution
