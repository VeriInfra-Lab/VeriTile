import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.HistoricalContract
import bench.repairs.tritonbench_g.matmul_triton_autotune.Kernels
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.Real
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.Wrapper

/-! Independent logical target: sum_k A[i,k]*B[k,j], followed by leaky ReLU
iff the exact activation string is "leaky_relu". roundHalf is the explicit
ideal model's final cast; id gives ideal Real GEMM. This theorem does not
replace the actual FP32 live dotAcc chain by a final rounding.
Domain: rank-two FP32 inputs on one supported device, contiguous A, arbitrary
valid nonnegative B strides, both negative-view flags allowed, matching inner
dimensions, fresh contiguous FP16 output, all three actual configurations.
Zero M/N/K, tails, A/B aliases, caller input preservation, output identity,
and every memory cell outside the output/copy allocation footprints are covered.
The tensor-library preparation contract is ordinary coordinate preservation;
there is no native-safe index-domain premise or final-output assumption. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
open VeriTile.Triton FP.Structural Whole
noncomputable section

specification matmul_view_safe_mathematical_correctness
    (roundHalf : ℝ → ℝ) (r : Request) (before prepared : State ℝ)
    (tensor : TensorAPI r prepared)
    (A : Fin r.rows → Fin r.inner → ℝ) (B : Fin r.inner → Fin r.columns → ℝ)
    (resolve : ResolveNegContract r before prepared tensor A B) :
    VeriTile.Spec.Real (∃ final,
      host (idealAlgebra roundHalf) idealDot r prepared=.ok (r.C,final) ∧
      (∀ i : Fin r.rows, ∀ j : Fin r.columns,
        final.mem r.C (outputOffset r i.val j.val)=Cell.mk .fp16 (roundHalf (mathematical r A B i j))) ∧
      (∀ region offset, ¬Footprint r region offset → final.mem region offset=prepared.mem region offset) ∧
      (∀ region offset, ¬Footprint r region offset →
        ¬CopyFootprint tensor resolve.copyA resolve.copyB region offset →
        final.mem region offset=before.mem region offset) ∧ AddressSafety r prepared tensor) :=
  resolved_mathematical roundHalf r before prepared tensor A B resolve

#axiomsClean matmul_view_safe_mathematical_correctness

/-- Full source-ordered wrapper result in the proved ideal interpretation.
Successful calls additionally expose the independent mathematical GEMM, while
all output identity, preparation trace and full caller frames remain explicit
in WrapperSuccess. Error outcomes do not claim a returned tensor. -/
specification matmul_view_safe_wrapper_mathematical_correctness
    (roundHalf : ℝ → ℝ) (api : WrapperAPI ℝ) (r : Request) (before : State ℝ)
    (A : Fin r.rows → Fin r.inner → ℝ) (B : Fin r.inner → Fin r.columns → ℝ)
    (apiLaw : WrapperAPIContract api r before A B) :
    VeriTile.Spec.Real (match wrapper (idealAlgebra roundHalf) idealDot api r before with
      | .error e => WrapperFailure api r before e
      | .ok (c,final) => WrapperSuccess (idealAlgebra roundHalf) idealDot (idealNumerics roundHalf) api r before A B c final ∧
        ∀ i : Fin r.rows, ∀ j : Fin r.columns,
          final.mem r.C (outputOffset r i.val j.val)=
            Cell.mk .fp16 (roundHalf (mathematical r A B i j))) := by
  have h := wrapper_correct (idealAlgebra roundHalf) idealDot (idealNumerics roundHalf)
    (ideal_primitives roundHalf) api r before A B apiLaw
  unfold WrapperOutcome at h
  cases hw : wrapper (idealAlgebra roundHalf) idealDot api r before with
  | error e => simpa only [hw] using h
  | ok output =>
    rcases output with ⟨c,final⟩
    simp only [hw] at h ⊢
    refine ⟨h,?_⟩
    obtain ⟨_,prepared,tensor,resolve,_,hvalues,_⟩ := h
    intro i j
    have href := reference_logical (idealNumerics roundHalf) r prepared A B
      resolve.logicalA resolve.logicalB i j
    have hmath := ideal_reference roundHalf r prepared A B resolve.logicalA resolve.logicalB i j
    have hv := hvalues i j
    rw [←href,hmath] at hv
    exact hv

#axiomsClean matmul_view_safe_wrapper_mathematical_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Checked
open VeriTile.Triton FP.Structural Whole
noncomputable section
specification checked_mathematical
    (roundHalf : ℝ → ℝ) (api : API ℝ) (p : Metadata) (before : State ℝ)
    (admitted : InputContract p) (library : Library api p before)
    (A : Fin p.request.rows → Fin p.request.inner → ℝ)
    (B : Fin p.request.inner → Fin p.request.columns → ℝ)
    (hA : ∀ i k,inputA p.request before i.val k.val=A i k)
    (hB : ∀ k j,inputB p.request before k.val j.val=B k j) :
    VeriTile.Spec.Real (match run (idealAlgebra roundHalf) idealDot api p before with
      | .error (.inl _) => False
      | .error (.inr e) => WrapperFailure api.legacy p.request before e
      | .ok (c,final) =>
        WrapperSuccess (idealAlgebra roundHalf) idealDot (idealNumerics roundHalf)
          api.legacy p.request before A B c final ∧
        (∀ region offset, ¬Footprint p.request region offset → final.mem region offset=before.mem region offset) ∧
        ∀ i : Fin p.request.rows, ∀ j : Fin p.request.columns,
          final.mem p.request.C (outputOffset p.request i.val j.val)=
            Cell.mk .fp16 (roundHalf (mathematical p.request A B i j))) := by
  have h := matmul_view_safe_wrapper_mathematical_correctness roundHalf api.legacy p.request before
    A B (library.legacy admitted A B hA hB)
  simp only [run,(checkInputs_iff p).2 admitted]
  cases he : wrapper (idealAlgebra roundHalf) idealDot api.legacy p.request before with
  | error e => simpa only [he,Except.mapError] using h
  | ok pair =>
    rcases pair with ⟨c,final⟩
    simp only [he] at h
    exact ⟨h.1,full_frame library h.1,h.2⟩
#axiomsClean checked_mathematical
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Checked
