import bench.tritonbench_g.kldiv_ops.Proofs.Mathematical_Kernels
import bench.tritonbench_g.geglu_tanh_triton.Proofs.Mathematical_FP

/-! The existing candidate commutes the exact signed output-row product only.
Every typed load, cast, floating expression, loop, branch and store is unchanged.
This is a structural equivalence, not a claim of a numerical optimization. -/
namespace VeriTile.Bench.TritonBenchG.KldivOps.FPProof
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

def rowPointer (v : Bool) (name : RegName) (R : RegionName) (stride : Nat) : ComputeStmt :=
  .assign .ptr [] name (.alg (.ptrAdd .nil (.ptrBase R) (.castIntToNat
    (if v then .mul .int .nil (.castNatToInt (.constNat stride)) (.ref .int [] "pid")
    else .mul .int .nil (.ref .int [] "pid") (.castNatToInt (.constNat stride))))))

def forwardPrefix (v : Bool) (Y T O : RegionName) (sy st so : Nat) : List ComputeStmt :=
  [.assign .int [] "pid" (.alg (.castNatToInt (.programId 0))),
   rowPointer Bool.false "Y" Y sy, rowPointer Bool.false "T" T st, rowPointer v "O" O so]

def backwardPrefix (v : Bool) (T O : RegionName) (st so : Nat) : List ComputeStmt :=
  [.assign .int [] "pid" (.alg (.castNatToInt (.programId 0))),
   rowPointer Bool.false "T" T st, rowPointer v "O" O so]

theorem rowPointer_step {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (name : RegName) (R : RegionName) (stride : Nat) :
    step (engine M) (rowPointer Bool.false name R stride) s =
      step (engine M) (rowPointer Bool.true name R stride) s := by
  have h := GegluTanhTriton.FPProof.signed_ptr_index_commutes M s [] R
    (.ref .int [] "pid") (.castNatToInt (.constNat stride)) s (by simp [run])
  simpa only [rowPointer, Bool.false_eq_true, ↓reduceIte, step, FP.Structural.evalExpr] using
    congrArg (fun x => x >>= fun value => some (s.setReg name .ptr [] value)) h

theorem forward_prefix_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (Y T O : RegionName) (sy st so : Nat) (after : List ComputeStmt) :
    run (engine M) (forwardPrefix Bool.false Y T O sy st so ++ after) s =
      run (engine M) (forwardPrefix Bool.true Y T O sy st so ++ after) s := by
  rw [run_append, run_append]
  apply congrArg (fun (result : Option (State α)) => result.bind (run (engine M) after))
  simp only [forwardPrefix, run, rowPointer_step]

theorem backward_prefix_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (T O : RegionName) (st so : Nat) (after : List ComputeStmt) :
    run (engine M) (backwardPrefix Bool.false T O st so ++ after) s =
      run (engine M) (backwardPrefix Bool.true T O st so ++ after) s := by
  rw [run_append, run_append]
  apply congrArg (fun (result : Option (State α)) => result.bind (run (engine M) after))
  simp only [backwardPrefix, run, rowPointer_step]

theorem forward_default_decomposition (pred target output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) :
    (forwardDefault pred target output v Y T O sy st so N B eps reduction).surfaceBody =
      forwardPrefix v Y T O sy st so ++
        (forwardDefault pred target output Bool.false Y T O sy st so N B eps reduction).surfaceBody.drop 4 := by
  cases pred <;> cases target <;> cases output <;> cases v <;> rfl

/-- Complete state and failure equality, for every numerical algebra and initial state. -/
theorem forward_default_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (pred target output : Storage) (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) (s : State α) :
    run (engine M) (forwardDefault pred target output Bool.false Y T O sy st so N B eps reduction).surfaceBody s =
      run (engine M) (forwardDefault pred target output Bool.true Y T O sy st so N B eps reduction).surfaceBody s := by
  rw [forward_default_decomposition pred target output Bool.false,
    forward_default_decomposition pred target output Bool.true]
  exact forward_prefix_run M s Y T O sy st so _

theorem forward_log_decomposition (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) :
    (forwardLog pred target output v Y T O sy st so N B eps reduction).surfaceBody =
      forwardPrefix v Y T O sy st so ++
        (forwardLog pred target output Bool.false Y T O sy st so N B eps reduction).surfaceBody.drop 4 := by
  cases pred <;> cases target <;> cases output <;> cases v <;> rfl

/-- Complete state and failure equality, for every numerical algebra and initial state. -/
theorem forward_log_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (pred : Storage) (target : MathStorage) (output : Storage) (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) (s : State α) :
    run (engine M) (forwardLog pred target output Bool.false Y T O sy st so N B eps reduction).surfaceBody s =
      run (engine M) (forwardLog pred target output Bool.true Y T O sy st so N B eps reduction).surfaceBody s := by
  rw [forward_log_decomposition pred target output Bool.false,
    forward_log_decomposition pred target output Bool.true]
  exact forward_prefix_run M s Y T O sy st so _

theorem backward_default_decomposition (target output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (backwardDefault target output v T O st so N B).surfaceBody =
      backwardPrefix v T O st so ++
        (backwardDefault target output Bool.false T O st so N B).surfaceBody.drop 3 := by
  cases target <;> cases output <;> cases v <;> rfl

/-- Complete state and failure equality, for every numerical algebra and initial state. -/
theorem backward_default_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (target output : Storage) (T O : RegionName) (st so N B : Nat) (s : State α) :
    run (engine M) (backwardDefault target output Bool.false T O st so N B).surfaceBody s =
      run (engine M) (backwardDefault target output Bool.true T O st so N B).surfaceBody s := by
  rw [backward_default_decomposition target output Bool.false,
    backward_default_decomposition target output Bool.true]
  exact backward_prefix_run M s T O st so _

theorem backward_log_decomposition (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (backwardLog target output v T O st so N B).surfaceBody =
      backwardPrefix v T O st so ++
        (backwardLog target output Bool.false T O st so N B).surfaceBody.drop 3 := by
  cases target <;> cases output <;> cases v <;> rfl

/-- Complete state and failure equality, for every numerical algebra and initial state. -/
theorem backward_log_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (target : MathStorage) (output : Storage) (T O : RegionName) (st so N B : Nat) (s : State α) :
    run (engine M) (backwardLog target output Bool.false T O st so N B).surfaceBody s =
      run (engine M) (backwardLog target output Bool.true T O st so N B).surfaceBody s := by
  rw [backward_log_decomposition target output Bool.false,
    backward_log_decomposition target output Bool.true]
  exact backward_prefix_run M s T O st so _

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivOps.FPProof
