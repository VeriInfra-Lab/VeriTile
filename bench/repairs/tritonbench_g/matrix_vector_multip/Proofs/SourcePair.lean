import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourceCode
import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeNative

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.SourcePair
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false

/-- Preserve the selected native model's existing fused carry interpretation.
This is not a numerical axiom equating separately rounded multiply/add to FMA. -/
def fusedAlgebra (native : Scalars) (g : Graph BN BM) : Algebra Payload :=
  { WholeNative.algebra native g with fma := fun _ => some (fun a b c => .carry c a b) }

variable (native : Scalars) (N M BN BM : Nat) (g : Graph BN BM)
  (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)

private theorem run_congr_steps {α : Type} [Inhabited α] (A B : Algebra α)
    {left right : List ComputeStmt}
    (same : List.Forall₂ (fun a b => ∀ s, step A a s = step B b s) left right)
    (s : State α) : run A left s = run B right s := by
  induction same generalizing s with
  | nil => simp only [run]
  | cons head tail ih => simp only [run,head,ih]

theorem prefix_steps :
    List.Forall₂ (fun a b => ∀ s, step (fusedAlgebra native g) a s =
      step (WholeNative.algebra native g) b s)
      (SourceCode.candidate N M BN BM rowShape columnShape).before
      (SourceCode.original N M BN BM rowShape columnShape).before := by
  simp only [SourceCode.candidate,SourceCode.original,Source.whileLoop?,Kernels.source,
    Kernels.explicitFMA,ComputeKernel.surfaceBody,Option.get_some,List.forall₂_cons]
  repeat' constructor
  all_goals
    intro s
    simp [step,evalExpr,evalComputeOp,evalOp_unfold,fusedAlgebra,WholeNative.algebra,
      numeric,bop,ofFloat,toFloat,ComputeDType.eraseDType,Option.bind,store,
      Broadcast.leftIndex,Broadcast.rightIndex]

theorem prefix_equal (s : State Payload) :
    run (fusedAlgebra native g) (SourceCode.candidate N M BN BM rowShape columnShape).before s =
      run (WholeNative.algebra native g) (SourceCode.original N M BN BM rowShape columnShape).before s :=
  run_congr_steps _ _ (prefix_steps native N M BN BM g rowShape columnShape) s

theorem body_steps :
    List.Forall₂ (fun a b => ∀ s, step (fusedAlgebra native g) a s =
      step (WholeNative.algebra native g) b s)
      (SourceCode.candidate N M BN BM rowShape columnShape).body
      (SourceCode.original N M BN BM rowShape columnShape).body := by
  simp only [SourceCode.candidate,SourceCode.original,Source.whileLoop?,Kernels.source,
    Kernels.explicitFMA,ComputeKernel.surfaceBody,Option.get_some,List.forall₂_cons]
  repeat' constructor
  all_goals
    intro s
    simp [step,evalExpr,evalComputeOp,evalOp_unfold,fusedAlgebra,WholeNative.algebra,
      numeric,bop,ofFloat,toFloat,ComputeDType.eraseDType,Option.bind,store,
      Broadcast.leftIndex,Broadcast.rightIndex]
  cases ha : s.regs .real [Random.i32ArangeSize BN rowShape,Random.i32ArangeSize BM columnShape] "a" <;>
    cases hb : s.regs .real [1,Random.i32ArangeSize BM columnShape] "b" <;>
    cases hc : s.regs .real [Random.i32ArangeSize BN rowShape,Random.i32ArangeSize BM columnShape] "acc" <;>
    simp [ha,hb,hc,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  congr 1

theorem body_equal (s : State Payload) :
    run (fusedAlgebra native g) (SourceCode.candidate N M BN BM rowShape columnShape).body s =
      run (WholeNative.algebra native g) (SourceCode.original N M BN BM rowShape columnShape).body s :=
  run_congr_steps _ _ (body_steps native N M BN BM g rowShape columnShape) s

theorem suffix_steps :
    List.Forall₂ (fun a b => ∀ s, step (fusedAlgebra native g) a s =
      step (WholeNative.algebra native g) b s)
      (SourceCode.candidate N M BN BM rowShape columnShape).after
      (SourceCode.original N M BN BM rowShape columnShape).after := by
  simp only [SourceCode.candidate,SourceCode.original,Source.whileLoop?,Kernels.source,
    Kernels.explicitFMA,ComputeKernel.surfaceBody,Option.get_some,List.forall₂_cons]
  repeat' constructor
  all_goals
    intro s
    simp [step,evalExpr,evalComputeOp,evalOp_unfold,fusedAlgebra,WholeNative.algebra,
      numeric,bop,ofFloat,toFloat,ComputeDType.eraseDType,Option.bind,store,
      Broadcast.leftIndex,Broadcast.rightIndex]

theorem suffix_equal (s : State Payload) :
    run (fusedAlgebra native g) (SourceCode.candidate N M BN BM rowShape columnShape).after s =
      run (WholeNative.algebra native g) (SourceCode.original N M BN BM rowShape columnShape).after s :=
  run_congr_steps _ _ (suffix_steps native N M BN BM g rowShape columnShape) s

theorem condition_equal (s : State Payload) :
    evalOp (fusedAlgebra native g) none (SourceCode.candidate N M BN BM rowShape columnShape).condition s =
      evalOp (WholeNative.algebra native g) none (SourceCode.original N M BN BM rowShape columnShape).condition s := by
  rfl

theorem loop_equal (s : State Payload) :
    step (fusedAlgebra native g)
      (.whileLoop (SourceCode.candidate N M BN BM rowShape columnShape).condition
        (SourceCode.candidate N M BN BM rowShape columnShape).body) s =
    step (WholeNative.algebra native g)
      (.whileLoop (SourceCode.original N M BN BM rowShape columnShape).condition
        (SourceCode.original N M BN BM rowShape columnShape).body) s := by
  simp only [step]
  have cond := funext (condition_equal native N M BN BM g rowShape columnShape)
  have body := funext (body_equal native N M BN BM g rowShape columnShape)
  simp only [cond, body]

/-- Equality of executions of the two complete raw bodies, including failure
and nontermination. The finite execution and IO proofs are separate. -/
theorem execution_equal (s : State Payload) :
    exec (fusedAlgebra native g) (Kernels.explicitFMA N M BN BM rowShape columnShape) s =
      exec (WholeNative.algebra native g) (Kernels.source N M BN BM rowShape columnShape) s := by
  change run (fusedAlgebra native g) (Kernels.explicitFMA N M BN BM rowShape columnShape).surfaceBody s =
    run (WholeNative.algebra native g) (Kernels.source N M BN BM rowShape columnShape).surfaceBody s
  rw [SourceCode.candidate_decomposition, SourceCode.original_decomposition]
  simp only [run_append, prefix_equal native N M BN BM g rowShape columnShape,
    run, loop_equal native N M BN BM g rowShape columnShape,
    suffix_equal native N M BN BM g rowShape columnShape, Option.bind_some, Option.bind_assoc]


end
end VeriTile.Bench.Repairs.MatrixVectorMultip.SourcePair
