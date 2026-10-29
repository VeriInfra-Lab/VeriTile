import bench.tritonbench_g.ksoftmax_triton.Kernels
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFullFP
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedValues
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedProfilesTotal
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardFP

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Equivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (gradient saved destination : Backward.Typed.Storage) (v : Bool)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) : Program :=
  Backward.Floating.fullProgram gradient saved destination v GradIn GradOut Out
    bm bn gm gn om on K D log causal fp16

/-- Whole typed state and failure equality for all independent storage triples
and flags. No floating arithmetic law or domain assumption is required. -/
specification ksoftmax_backward_index_equiv
    (gradient saved destination : Backward.Typed.Storage)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) :
    program gradient saved destination Bool.false GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16 ≡[assumptions]
    program gradient saved destination Bool.true GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16 :=
  Backward.Floating.backward_full_index_equiv gradient saved destination
    GradIn GradOut Out bm bn gm gn om on K D log causal fp16

/-- A common successful final state for each of six homogeneous storage
profiles and every flag combination, in the explicit exact-real algebra.
Registers start empty; program IDs, strides and masked tails are nontrivial.
This witnesses formal execution and does not assert native IEEE admission. -/
theorem ksoftmax_backward_typed_nonempty (storage : Backward.Typed.Storage)
    (log causal fp16 : Bool) :
    ∃ final, FP.Structural.exec (engine Backward.TypedExecution.model)
        (program storage storage storage Bool.false "D" "G" "O"
          11 7 13 5 17 3 3 4 log causal fp16).kernel
        (Backward.TypedExecution.initial storage log) = some final ∧
      FP.Structural.exec (engine Backward.TypedExecution.model)
        (program storage storage storage Bool.true "D" "G" "O"
          11 7 13 5 17 3 3 4 log causal fp16).kernel
        (Backward.TypedExecution.initial storage log) = some final := by
  have hExec (k : ComputeKernel) (s : FP.Structural.State (WithBot ℝ)) :
      FP.Structural.exec (engine Backward.TypedExecution.model) k s =
      FP.Structural.run (engine Backward.TypedExecution.model) k.surfaceBody s := by
    cases k <;> rfl
  simp only [program, Backward.Floating.fullProgram, hExec,
    Backward.Typed.homogeneous_agrees]
  exact ⟨(Backward.TypedExecution.source_run storage log causal fp16 Bool.false).val,
    (Backward.TypedExecution.source_run storage log causal fp16 Bool.false).property,
    Backward.TypedExecution.candidate_runs storage log causal fp16⟩

/-- Both complete forward variants reach a common typed final state for
the four checked profiles, absent/qk/bk masks and all source flags.
Positive depth is required; algebra, initial state, regions and strides
remain arbitrary. This establishes structural execution success, without
a numerical-refinement or native-admission claim. -/
theorem ksoftmax_forward_typed_nonempty
    {α : Type} [Inhabited α] (algebra : FP.Structural.Algebra α)
    (profile : Forward.Typed.CheckedProfile)
    (Y X M : RegionName) (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : Forward.MaskMode) (causal fp16 : Bool)
    (initial : FP.Structural.State α) :
    ∃ final : FP.Structural.State α, ∀ commuted : Bool,
      FP.Structural.run (engine algebra)
        (Forward.Typed.checkedKernel profile commuted Y X M ym yn xm xn sm K D
          log mode.python causal fp16).surfaceBody initial = some final :=
  Forward.TypedTotal.checked_nonempty_pair algebra profile
    Y X M ym yn xm xn sm K D hD log mode causal fp16 initial

#axiomsClean ksoftmax_forward_typed_nonempty

#print_fp_assumptions ksoftmax_backward_index_equiv
#axiomsClean ksoftmax_backward_index_equiv
#axiomsClean ksoftmax_backward_typed_nonempty
/-- Complete typed original/candidate equivalence for the four checked
source profiles. Forward.Floating.assumptions is the empty list. -/
specification ksoftmax_forward_index_equiv
    (profile : Forward.Typed.CheckedProfile)
    (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log : Bool) (mode : Option String) (causal fp16 : Bool) :
    Forward.Floating.program profile Bool.false Y X M ym yn xm xn sm K D
      log mode causal fp16 ≡[Forward.Floating.assumptions]
    Forward.Floating.program profile Bool.true Y X M ym yn xm xn sm K D
      log mode causal fp16 :=
  Forward.Floating.forward_index_equiv profile Y X M ym yn xm xn sm K D
    log mode causal fp16

/-- Equality of actual typed interpreter runs, including failures, for
arbitrary initial state and algebra within the same four-profile scope. -/
theorem ksoftmax_forward_typed_run_eq
    (profile : Forward.Typed.CheckedProfile)
    (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log : Bool) (mode : Option String) (causal fp16 : Bool)
    {α : Type} [Inhabited α] (algebra : Algebra α) (initial : State α) :
    run (engine algebra)
      (Forward.Typed.checkedKernel profile Bool.false Y X M ym yn xm xn sm K D
        log mode causal fp16).surfaceBody initial =
    run (engine algebra)
      (Forward.Typed.checkedKernel profile Bool.true Y X M ym yn xm xn sm K D
        log mode causal fp16).surfaceBody initial :=
  Forward.Floating.forward_run_eq profile Y X M ym yn xm xn sm K D
    log mode causal fp16 algebra initial

#print_fp_assumptions ksoftmax_forward_index_equiv
#axiomsClean ksoftmax_forward_index_equiv
#axiomsClean ksoftmax_forward_typed_run_eq

end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Equivalence
