import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.HistoricalContract
import bench.repairs.tritonbench_g.matmul_triton_autotune.Kernels
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.FP
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.Wrapper

/-! Complete FP32 host/grid contract against the independent logical recurrence.
Each configuration retains BK=64/32/16, including padded zero lanes and the
incoming FP32 accumulator in the same dotAcc primitive. Comparisons and 0.01*x
remain separate scalar primitives; fixed FP16 conversion/store stays explicit.
No FP=Real, FMA=separate multiply/add, or cross-configuration equality is claimed.

Numerical domain is the full scalar domain of M/D/Q, including reachable input,
masked-zero, carry, activation and conversion values. PrimitiveLaws has exactly
six local correspondences: initialZero, loadZero, dotAcc, comparison, slope,
halfCast. The dot profile is source-default FP32 (including target-default
TF32 behavior where applicable); native conformance for the selected target is
not established by this theorem. No execution, shape, index or frame law is
assumed numerically. Logical views use ResolveNegContract instead.
-/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
open VeriTile.Triton FP.Structural Whole
noncomputable section

/-- Whole typed host execution and return identity require no numerical ports
and no assumed successful execution. Tensor preparation is an ordinary API
boundary; this theorem proves the subsequent candidate program succeeds. -/
specification matmul_view_safe_whole_execution {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (r : Request) (prepared : State α)
    (dimensions : r.inner=r.innerB) (contiguousA : r.aContiguous=Bool.true) :
    ∃ final, host M D r prepared=.ok (r.C,final) :=
  host_executes M D r prepared dimensions contiguousA

#axiomsClean matmul_view_safe_whole_execution

specification matmul_view_safe_fp32_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (before prepared : State α) (tensor : TensorAPI r prepared)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (resolve : ResolveNegContract r before prepared tensor A B) :
    ∃ final, host M D r prepared=.ok (r.C,final) ∧
      (∀ i : Fin r.rows, ∀ j : Fin r.columns,
        final.mem r.C (outputOffset r i.val j.val)=Cell.mk .fp16 (logicalReference Q r A B i j)) ∧
      (∀ region offset, ¬Footprint r region offset → final.mem region offset=prepared.mem region offset) ∧
      (∀ region offset, ¬Footprint r region offset →
        ¬CopyFootprint tensor resolve.copyA resolve.copyB region offset →
        final.mem region offset=before.mem region offset) ∧ AddressSafety r prepared tensor :=
  resolved_fp M D Q law r before prepared tensor A B resolve

#axiomsClean matmul_view_safe_fp32_correctness

/-- Public total wrapper execution, including source-ordered API errors.
The earlier whole_execution entry is the internal post-preparation result. -/
specification matmul_view_safe_wrapper_execution {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α)
    (api : WrapperAPI α) (r : Request) (before : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (apiLaw : WrapperAPIContract api r before A B) :
    match wrapper M D api r before with
    | .error e => WrapperFailure api r before e
    | .ok (c,_) => c=r.C := wrapper_executes M D api r before A B apiLaw

/-- Public full pre-host-to-final FP result. No setup/launch success premise;
the success branch contains all output and frame facts, while the error branch
identifies the actual first reached source stage. -/
specification matmul_view_safe_wrapper_fp32_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (api : WrapperAPI α) (r : Request) (before : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (apiLaw : WrapperAPIContract api r before A B) :
    WrapperOutcome M D Q api r before A B := wrapper_correct M D Q law api r before A B apiLaw

#axiomsClean matmul_view_safe_wrapper_execution
#axiomsClean matmul_view_safe_wrapper_fp32_correctness
-- This is a conditional host-result theorem, not a Spec.FP derivation;
-- print_fp_assumptions rejects this type. Print the exact local interface.
#print PrimitiveLaws
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Checked
open VeriTile.Triton FP.Structural Whole
noncomputable section
specification checked_floating {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (api : API α) (p : Metadata) (before : State α) (admitted : InputContract p)
    (library : Library api p before)
    (A : Fin p.request.rows → Fin p.request.inner → α)
    (B : Fin p.request.inner → Fin p.request.columns → α)
    (hA : ∀ i k,inputA p.request before i.val k.val=A i k)
    (hB : ∀ k j,inputB p.request before k.val j.val=B k j) :
    match run M D api p before with
    | .error (.inl _) => False
    | .error (.inr e) => WrapperFailure api.legacy p.request before e
    | .ok (c,final) => WrapperSuccess M D Q api.legacy p.request before A B c final ∧
      ∀ region offset, ¬Footprint p.request region offset → final.mem region offset=before.mem region offset := by
  have h := matmul_view_safe_wrapper_fp32_correctness M D Q law api.legacy p.request before
    A B (library.legacy admitted A B hA hB)
  unfold WrapperOutcome at h
  simp only [run,(checkInputs_iff p).2 admitted]
  cases he : wrapper M D api.legacy p.request before with
  | error e => simpa only [he,Except.mapError] using h
  | ok pair =>
    rcases pair with ⟨c,final⟩
    simp only [he] at h
    exact ⟨h,full_frame library h⟩
#axiomsClean checked_floating
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Checked
