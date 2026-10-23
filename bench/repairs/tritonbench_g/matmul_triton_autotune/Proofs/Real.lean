import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.Host
import bench.tritonbench_g.matmul_triton_autotune.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
open VeriTile.Triton FP.Structural Whole
noncomputable section

theorem resolved_mathematical (roundHalf : ℝ → ℝ) (r : Request)
    (before prepared : State ℝ) (tensor : TensorAPI r prepared)
    (A : Fin r.rows → Fin r.inner → ℝ) (B : Fin r.inner → Fin r.columns → ℝ)
    (resolve : ResolveNegContract r before prepared tensor A B) :
    ∃ final, host (idealAlgebra roundHalf) idealDot r prepared=.ok (r.C,final) ∧
      (∀ i : Fin r.rows, ∀ j : Fin r.columns,
        final.mem r.C (outputOffset r i.val j.val)=Cell.mk .fp16 (roundHalf (mathematical r A B i j))) ∧
      (∀ region offset, ¬Footprint r region offset → final.mem region offset=prepared.mem region offset) ∧
      (∀ region offset, ¬Footprint r region offset →
        ¬CopyFootprint tensor resolve.copyA resolve.copyB region offset →
        final.mem region offset=before.mem region offset) ∧ AddressSafety r prepared tensor := by
  obtain ⟨final,he,⟨hv,hf⟩,hc,hs⟩ := resolved_host
    (idealAlgebra roundHalf) idealDot (idealNumerics roundHalf) (ideal_primitives roundHalf)
    r before prepared tensor A B resolve
  refine ⟨final,he,?_,hf,hc,hs⟩
  intro i j
  simpa only [ideal_reference roundHalf r prepared A B resolve.logicalA resolve.logicalB i j]
    using hv i.val i.isLt j.val j.isLt

#axiomsClean resolved_mathematical
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
