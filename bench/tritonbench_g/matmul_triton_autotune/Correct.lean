import bench.tritonbench_g.matmul_triton_autotune.Kernels
import bench.tritonbench_g.matmul_triton_autotune.Proofs.Host
import bench.tritonbench_g.matmul_triton_autotune.Proofs.Real

/-! Independent ideal activated GEMM, all actual configs and ragged K.
Arithmetic here uses an explicit ideal Real interpretation; roundHalf is only
the final boundary cast. This is not an IEEE/GPU equivalence claim. The separate
legacy whole-K Real/rounding-model theorems keep their original statements. -/
/-! Scope: hA/hB pin the independent arrays to physical storage values. The
original wrapper does not resolve Torch lazy-negative flags; these premises do
not cover its logical values for such views. Proofs/LogicalViewDefect.lean and
the task report establish that source defect. No false-flag restriction is
silently added, and these raw-storage theorems earn no whole-task acceptance. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Whole
open VeriTile.Triton FP.Structural
noncomputable section

specification matmul_autotune_raw_storage_mathematical_host_correctness
    (roundHalf : ℝ → ℝ) (r : Request) (initial : State ℝ)
    (A : Fin r.rows → Fin r.inner → ℝ) (B : Fin r.inner → Fin r.columns → ℝ)
    (hA : ∀ i k,inputA r initial i.val k.val=A i k)
    (hB : ∀ k j,inputB r initial k.val j.val=B k j)
    (tensor : TensorContract r initial) (indices : IndexDomain r) :
    VeriTile.Spec.Real (∃ final, host (idealAlgebra roundHalf) idealDot r initial=.ok (r.C,final) ∧
      (∀ i : Fin r.rows, ∀ j : Fin r.columns,
        final.mem r.C (outputOffset r i.val j.val)=Cell.mk .fp16 (roundHalf (mathematical r A B i j))) ∧
      (∀ region offset, ¬Footprint r region offset → final.mem region offset=initial.mem region offset) ∧
      SourceSafety r initial tensor) := by
  obtain ⟨final,he,hv,hf⟩ := host_correct (idealAlgebra roundHalf) idealDot (idealNumerics roundHalf)
    (ideal_primitives roundHalf) r initial tensor.freshA tensor.freshB tensor.dimensions tensor.contiguousA
  refine ⟨final,he,?_,hf,source_safety r initial tensor indices⟩
  intro i j
  simpa only [ideal_reference roundHalf r initial A B hA hB i j] using hv i.val i.isLt j.val j.isLt

#axiomsClean matmul_autotune_raw_storage_mathematical_host_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Whole
