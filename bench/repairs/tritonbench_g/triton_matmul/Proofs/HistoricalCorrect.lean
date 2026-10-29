import bench.repairs.tritonbench_g.triton_matmul.Kernels
import bench.repairs.tritonbench_g.triton_matmul.Proofs.Real
/-! Independent logical Real GEMM; internal casts/rounding are erased by the explicit ideal model. Actual storage labels and all program steps remain.
Domain: rank-two matching FP16/E4M3FN inputs on one supported device, arbitrary valid nonnegative strides for both A/B, all logical-negative input flags, zero dimensions/tails/aliases, fresh contiguous same-type C. Tensor preparation is the ordinary coordinate-preserving library contract. Full output, returned C, caller/prepared frames and both original input frames are explicit below. -/
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
open VeriTile.Triton FP.Structural FPValue
noncomputable section

specification matmul_repaired_mathematical_correctness (r : Request) (before prepared : State ℝ) (tensor : TensorAPI r prepared)
    (A : Fin r.rows → Fin r.inner → ℝ) (B : Fin r.inner → Fin r.columns → ℝ)
    (resolve : ResolveNegContract r before prepared tensor A B) :
    VeriTile.Spec.Real (∃ final, host ideal idealDot r prepared=.ok (r.C,final) ∧
      (∀ i : Fin r.rows, ∀ j : Fin r.columns,
        final.mem r.C (outputOffset r i.val j.val)=stored r.storage (SourceScope.mathematical r.rows r.columns r.inner A B i j)) ∧
      (∀ region offset, ¬Footprint r region offset → final.mem region offset=prepared.mem region offset) ∧
      (∀ region offset, ¬Footprint r region offset → ¬CopyFootprint tensor resolve.copyA resolve.copyB region offset →
        final.mem region offset=before.mem region offset) ∧
      (∀ offset, final.mem resolve.originalA offset=before.mem resolve.originalA offset) ∧
      (∀ offset, final.mem resolve.originalB offset=before.mem resolve.originalB offset) ∧
      AddressSafety r prepared tensor) :=
  resolved_mathematical r before prepared tensor A B resolve

/-- Whole source-ordered wrapper. The Real target is independent logical GEMM;
every success carries both original input frames and finite source execution. -/
specification matmul_repaired_wrapper_mathematical_correctness
    (api : WrapperAPI ℝ) (r : Request) (before : State ℝ)
    (A : Fin r.rows → Fin r.inner → ℝ) (B : Fin r.inner → Fin r.columns → ℝ)
    (apiLaw : WrapperAPIContract api r before A B) :
    VeriTile.Spec.Real (match wrapper ideal idealDot api r before with
    | .error e => WrapperFailure api r before e
    | .ok (c,final) => WrapperSuccess idealNumerics api r before A B c final ∧
        (∀ i : Fin r.rows, ∀ j : Fin r.columns,
          final.mem r.C (outputOffset r i.val j.val)=stored r.storage (SourceScope.mathematical r.rows r.columns r.inner A B i j))) :=
  wrapper_mathematical api r before A B apiLaw

#axiomsClean matmul_repaired_wrapper_mathematical_correctness

#axiomsClean matmul_repaired_mathematical_correctness

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
