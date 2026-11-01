import bench.tritonbench_g.kldiv_ops.Proofs.Mathematical_Kernels
import bench.tritonbench_g.geglu_tanh_triton.Proofs.Mathematical_FP

/-! The existing candidate commutes the exact signed output-row product only.
Every typed load, cast, floating expression, loop, branch and store is unchanged.
This is a structural equivalence, not a claim of a numerical optimization. -/
namespace VeriTile.Bench.TritonBenchG.KldivOps.FPProof
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

def integerArgument (name : RegName) (n : Nat) : ComputeStmt :=
  .assign .int [] name (.alg (.castNatToInt (.constNat n)))

def rowPointer (v : Bool) (name : RegName) (R : RegionName) (stride : Nat) : ComputeStmt :=
  .assign .ptr [] name (.alg (.ptrAdd .nil (.ptrBase R) (.castIntToNat
    (.add .int .nil (.castNatToInt (.constNat 0))
      (if v then .mul .int .nil (.castNatToInt (.constNat stride)) (.ref .int [] "pid")
       else .mul .int .nil (.ref .int [] "pid") (.castNatToInt (.constNat stride)))))))

def forwardPrefix (v : Bool) (Y T O : RegionName) (sy st so N : Nat) : List ComputeStmt :=
  [integerArgument "n_cols" N,integerArgument "y_stride" sy,integerArgument "gt_stride" st,
   integerArgument "loss_stride" so,.assign .int [] "pid" (.alg (.castNatToInt (.programId 0))),
   rowPointer Bool.false "y_ptr" Y sy,rowPointer Bool.false "gt_ptr" T st,rowPointer v "loss_ptr" O so]

def backwardPrefix (v : Bool) (T O : RegionName) (st so N : Nat) : List ComputeStmt :=
  [integerArgument "n_cols" N,integerArgument "target_stride" st,integerArgument "new_grads_stride" so,
   .assign .int [] "pid" (.alg (.castNatToInt (.programId 0))),
   rowPointer Bool.false "target_ptr" T st,rowPointer v "new_grads_ptr" O so]

theorem rowPointer_step {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (name : RegName) (R : RegionName) (stride : Nat) :
    step (engine M) (rowPointer Bool.false name R stride) s =
      step (engine M) (rowPointer Bool.true name R stride) s := by
  cases hp : s.regs .int [] "pid"
  all_goals simp [rowPointer,step,evalExpr,evalOp_unfold,Option.bind,numeric,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,hp,Int.mul_comm]

theorem forward_prefix_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (Y T O : RegionName) (sy st so N : Nat) (after : List ComputeStmt) :
    run (engine M) (forwardPrefix Bool.false Y T O sy st so N ++ after) s =
      run (engine M) (forwardPrefix Bool.true Y T O sy st so N ++ after) s := by
  rw [run_append, run_append]
  apply congrArg (fun (result : Option (State α)) => result.bind (run (engine M) after))
  simp only [forwardPrefix, run, rowPointer_step]

theorem backward_prefix_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (T O : RegionName) (st so N : Nat) (after : List ComputeStmt) :
    run (engine M) (backwardPrefix Bool.false T O st so N ++ after) s =
      run (engine M) (backwardPrefix Bool.true T O st so N ++ after) s := by
  rw [run_append, run_append]
  apply congrArg (fun (result : Option (State α)) => result.bind (run (engine M) after))
  simp only [backwardPrefix, run, rowPointer_step]

theorem backward_default_decomposition (target output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (backwardDefault target output v T O st so N B).surfaceBody =
      backwardPrefix v T O st so N ++
        (backwardDefault target output Bool.false T O st so N B).surfaceBody.drop 6 := by
  cases target <;> cases output <;> cases v <;> rfl

/-- Complete state and failure equality, for every numerical algebra and initial state. -/
theorem backward_default_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (target output : Storage) (T O : RegionName) (st so N B : Nat) (s : State α) :
    run (engine M) (backwardDefault target output Bool.false T O st so N B).surfaceBody s =
      run (engine M) (backwardDefault target output Bool.true T O st so N B).surfaceBody s := by
  rw [backward_default_decomposition target output Bool.false,
    backward_default_decomposition target output Bool.true]
  exact backward_prefix_run M s T O st so N _

theorem backward_log_decomposition (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (backwardLog target output v T O st so N B).surfaceBody =
      backwardPrefix v T O st so N ++
        (backwardLog target output Bool.false T O st so N B).surfaceBody.drop 6 := by
  cases target <;> cases output <;> cases v <;> rfl

/-- Complete state and failure equality, for every numerical algebra and initial state. -/
theorem backward_log_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (target : MathStorage) (output : Storage) (T O : RegionName) (st so N B : Nat) (s : State α) :
    run (engine M) (backwardLog target output Bool.false T O st so N B).surfaceBody s =
      run (engine M) (backwardLog target output Bool.true T O st so N B).surfaceBody s := by
  rw [backward_log_decomposition target output Bool.false,
    backward_log_decomposition target output Bool.true]
  exact backward_prefix_run M s T O st so N _

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivOps.FPProof
