import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardConstructors
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFP

/-! Structural equivalence of complete forward original/candidate programs.
BackwardFP supplies only the already checked natural multiplication and
pointer-add congruence rules. Every floating operation and control path is
the shared source suffix; no floating arithmetic atom is assumed.

This is a draft for root validation against the staged provenance frontend.
It makes no all-profile elaboration, successful-run, or native IEEE claim.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.Floating
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (profile : Typed.CheckedProfile) (commuted : Bool)
    (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log : Bool) (mode : Option String) (causal fp16 : Bool) : Program where
  kernel := Typed.checkedKernel profile commuted Y X M ym yn xm xn sm K D
    log mode causal fp16
  domain := []

/-- Full typed state and failure equality for the four declared source
profiles, with arbitrary dimensions, strides, regions, flags and mask option.
In particular no assumption removes the unknown-string failure path. -/
specification forward_index_equiv (profile : Typed.CheckedProfile)
    (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log : Bool) (mode : Option String) (causal fp16 : Bool) :
    program profile Bool.false Y X M ym yn xm xn sm K D log mode causal fp16
      ≡[assumptions]
    program profile Bool.true Y X M ym yn xm xn sm K D log mode causal fp16 := by
  cases profile <;> equiv_decompose <;> fp_prove

/-- Expose the same result directly as equality of actual interpreter runs,
without requiring callers to supply a successful execution or run equality. -/
theorem forward_run_eq (profile : Typed.CheckedProfile)
    (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log : Bool) (mode : Option String) (causal fp16 : Bool)
    {α : Type} [Inhabited α] (algebra : Algebra α) (initial : State α) :
    run (engine algebra)
      (Typed.checkedKernel profile Bool.false Y X M ym yn xm xn sm K D
        log mode causal fp16).surfaceBody initial =
    run (engine algebra)
      (Typed.checkedKernel profile Bool.true Y X M ym yn xm xn sm K D
        log mode causal fp16).surfaceBody initial := by
  let original := (Typed.checkedKernel profile Bool.false Y X M ym yn xm xn sm K D
    log mode causal fp16).surfaceBody
  let candidate := (Typed.checkedKernel profile Bool.true Y X M ym yn xm xn sm K D
    log mode causal fp16).surfaceBody
  -- Only the fourth assignment, `x_ptrs`, differs. Retain its full source context.
  have h : Contextual algebra initial (original.take 3) (original.drop 4)
      (original.getD 3 (.ifThen (.constBool Bool.false) []))
      (candidate.getD 3 (.ifThen (.constBool Bool.false) [])) := by
    cases profile <;> fp_prove
  cases profile <;> exact h

#print_fp_assumptions forward_index_equiv
#axiomsClean forward_index_equiv
#axiomsClean forward_run_eq
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.Floating
