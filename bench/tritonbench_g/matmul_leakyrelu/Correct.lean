import bench.tritonbench_g.matmul_leakyrelu.Proofs.RealModel

namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

/-- Complete mathematical output and exact full frame from the same shared
source as the floating theorem. The input functions are independent logical
matrix values, explicitly pinned to the initial tensor views. -/
specification matmul_leakyrelu_mathematical_host_correctness
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
  refine ⟨final,he,?_,hf,source_safety r indices initial tensor⟩
  intro i j
  simpa only [ideal_reference roundHalf r initial A B hA hB i j] using hv i.val i.isLt j.val j.isLt

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole


namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole.Contracted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

/-- Mathematical projection of the same contracted kernel used by the final
compiled-profile floating entry; roundHalf=id recovers the exact Real target. -/
specification matmul_leakyrelu_contracted_mathematical_host_correctness
    (roundHalf : ℝ → ℝ) (r : Request) (initial : State ℝ)
    (A : Fin r.rows → Fin r.inner → ℝ) (B : Fin r.inner → Fin r.columns → ℝ)
    (hA : ∀ i k,inputA r initial i.val k.val=A i k)
    (hB : ∀ k j,inputB r initial k.val j.val=B k j)
    (tensor : TensorContract r initial) (indices : IndexDomain r) :
    VeriTile.Spec.Real (∃ final, host (Whole.idealAlgebra roundHalf) idealDotAcc r initial=.ok (r.C,final) ∧
      (∀ i : Fin r.rows, ∀ j : Fin r.columns,
        final.mem r.C (outputOffset r i.val j.val)=Cell.mk .fp16 (roundHalf (mathematical r A B i j))) ∧
      (∀ region offset, ¬Footprint r region offset → final.mem region offset=initial.mem region offset) ∧
      SourceSafety r initial tensor) := by
  obtain ⟨final,he,hv,hf⟩ := host_correct (Whole.idealAlgebra roundHalf) idealDotAcc (idealNumerics roundHalf)
    (ideal_primitives roundHalf) r initial tensor.freshA tensor.freshB tensor.dimensions tensor.contiguousA
  refine ⟨final,he,?_,hf,source_safety r indices initial tensor⟩
  intro i j
  simpa only [ideal_reference roundHalf r initial A B hA hB i j] using hv i.val i.isLt j.val j.isLt

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole.Contracted
