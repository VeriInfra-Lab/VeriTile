import bench.repairs.tritonbench_g.kldiv_ops.IO
import bench.tritonbench_g.geglu_tanh_triton.Proofs.FP
import VeriTile.Triton.Float.StructuralSafety

/-! The full original and candidate differ only at one signed output-row
pointer product. All casts, branches, loads and floating expressions remain. -/
namespace VeriTile.Bench.Repairs.KldivOps.SourcePair
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def rowPointer (candidate : Bool) (name : RegName) (region : RegionName) (stride : RegName) : ComputeStmt :=
  .assign .ptr [] name (.alg (.ptrAdd .nil (.ptrBase region) (.castIntToNat
    (if candidate then .mul .int .nil (.ref .int [] stride) (.ref .int [] "pid")
     else .mul .int .nil (.ref .int [] "pid") (.ref .int [] stride)))))

theorem rowPointer_step (M : Algebra α) (s : State α) (name : RegName)
    (region : RegionName) (stride : RegName) :
    step (engine M) (rowPointer Bool.true name region stride) s =
      step (engine M) (rowPointer Bool.false name region stride) s := by
  have h := VeriTile.Bench.TritonBenchG.GegluTanhTriton.FPProof.signed_ptr_index_commutes
    M s [] region (.ref .int [] "pid") (.ref .int [] stride) s (by simp [run])
  symm
  simpa only [rowPointer, Bool.false_eq_true, ↓reduceIte, step, evalExpr] using
    congrArg (fun x => x >>= fun value => some (s.setReg name .ptr [] value)) h

theorem context_run (M : Algebra α) (before after : List ComputeStmt)
    (name : RegName) (region : RegionName) (stride : RegName) (s : State α) :
    run (engine M) (before ++ rowPointer Bool.true name region stride :: after) s =
      run (engine M) (before ++ rowPointer Bool.false name region stride :: after) s := by
  simp only [run_append,run,rowPointer_step]

theorem context_safe (M : Algebra α) (bounds : RegionName → Nat) (before after : List ComputeStmt)
    (name : RegName) (region : RegionName) (stride : RegName) (s : State α) :
    traceSafeList (engine M) bounds (before ++ rowPointer Bool.true name region stride :: after) s ↔
      traceSafeList (engine M) bounds (before ++ rowPointer Bool.false name region stride :: after) s := by
  simp only [traceSafeList_append,traceSafeList_cons,rowPointer_step]
  simp only [rowPointer,Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    stepSafe_unfold,safeExpr_unfold,safeOp_unfold,true_and,and_true]

theorem forward_decomposition (p : Parameters) (flag candidate : Bool) (red : Nat) :
    (if candidate then KernelDSL.forwardCandidate p flag red else KernelDSL.forward p flag red).surfaceBody =
      (KernelDSL.forward p flag red).surfaceBody.take 7 ++
      rowPointer candidate "loss_ptr" p.O "loss_stride" ::
      (KernelDSL.forward p flag red).surfaceBody.drop 8 := by
  cases flag <;> cases candidate <;> rfl

theorem backward_decomposition (p : Parameters) (flag candidate : Bool) :
    (if candidate then KernelDSL.backwardCandidate p flag else KernelDSL.backward p flag).surfaceBody =
      (KernelDSL.backward p flag).surfaceBody.take 5 ++
      rowPointer candidate "new_grads_ptr" p.O "new_grads_stride" ::
      (KernelDSL.backward p flag).surfaceBody.drop 6 := by
  cases flag <;> cases candidate <;> rfl

theorem forward_run (M : Algebra α) (p : Parameters) (flag : Bool) (red : Nat) (s : State α) :
    run (engine M) (KernelDSL.forwardCandidate p flag red).surfaceBody s =
      run (engine M) (KernelDSL.forward p flag red).surfaceBody s := by
  conv_lhs => rw [show (KernelDSL.forwardCandidate p flag red).surfaceBody = _ from
    forward_decomposition p flag Bool.true red]
  conv_rhs => rw [show (KernelDSL.forward p flag red).surfaceBody = _ from
    forward_decomposition p flag Bool.false red]
  exact context_run M _ _ _ _ _ s

theorem backward_run (M : Algebra α) (p : Parameters) (flag : Bool) (s : State α) :
    run (engine M) (KernelDSL.backwardCandidate p flag).surfaceBody s =
      run (engine M) (KernelDSL.backward p flag).surfaceBody s := by
  conv_lhs => rw [show (KernelDSL.backwardCandidate p flag).surfaceBody = _ from
    backward_decomposition p flag Bool.true]
  conv_rhs => rw [show (KernelDSL.backward p flag).surfaceBody = _ from
    backward_decomposition p flag Bool.false]
  exact context_run M _ _ _ _ _ s

theorem forward_safe (M : Algebra α) (p : Parameters) (flag : Bool) (red : Nat)
    (bounds : RegionName → Nat) (s : State α) :
    TraceSafe (engine M) bounds (KernelDSL.forwardCandidate p flag red) s ↔
      TraceSafe (engine M) bounds (KernelDSL.forward p flag red) s := by
  unfold TraceSafe
  conv_lhs => rw [show (KernelDSL.forwardCandidate p flag red).surfaceBody = _ from
    forward_decomposition p flag Bool.true red]
  conv_rhs => rw [show (KernelDSL.forward p flag red).surfaceBody = _ from
    forward_decomposition p flag Bool.false red]
  exact context_safe M bounds _ _ _ _ _ s

theorem backward_safe (M : Algebra α) (p : Parameters) (flag : Bool)
    (bounds : RegionName → Nat) (s : State α) :
    TraceSafe (engine M) bounds (KernelDSL.backwardCandidate p flag) s ↔
      TraceSafe (engine M) bounds (KernelDSL.backward p flag) s := by
  unfold TraceSafe
  conv_lhs => rw [show (KernelDSL.backwardCandidate p flag).surfaceBody = _ from
    backward_decomposition p flag Bool.true]
  conv_rhs => rw [show (KernelDSL.backward p flag).surfaceBody = _ from
    backward_decomposition p flag Bool.false]
  exact context_safe M bounds _ _ _ _ _ s

#axiomsClean forward_run
#axiomsClean backward_run
#axiomsClean forward_safe
#axiomsClean backward_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.SourcePair
