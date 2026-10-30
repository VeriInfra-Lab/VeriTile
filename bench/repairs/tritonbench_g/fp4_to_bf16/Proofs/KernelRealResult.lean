import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.RawKernel
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.Fp4ToBf16.KernelRealResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host HostCorrect RealSpec
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Every raw packed byte is decoded high nibble first to the independent
E2M1 Real value; complete finite source execution and memory frame are included. -/
theorem fp4_to_bf16_real (rest : Algebra Word) (h : Input) (storage : Finite.Storage h.shape.count)
    (jobs : List Nat) (initial : State Word) (fresh : h.input≠h.output)
    (typed : PackedInput h initial) (gridSafe : grid h≤2147483647)
    (order : jobs.Perm (List.range (grid h))) :
    VeriTile.Spec.Real (∃ final, Launch.run rest h jobs initial=some final ∧
      (∀ j,j<h.shape.count*2 →
        finiteReal (readBf16 (final.mem h.output j))=some (e2m1 (nibble h initial j))) ∧
      (∀ r o,r≠h.output ∨ h.shape.count*2≤o → final.mem r o=initial.mem r o)) := by
  obtain ⟨final,run,memory,frame⟩ := RawKernel.complete rest h storage jobs initial fresh typed gridSafe order
  refine ⟨final,run,?_,frame⟩
  intro j hj
  rw [memory]
  simp only [expectedMemory,hj,and_self,if_true]
  rw [read_cell]
  exact expected_real _

#axiomsClean fp4_to_bf16_real
#specNonCircular e2m1 avoiding [source,Kernels.source,Launch.run]
end
end VeriTile.Bench.Repairs.Fp4ToBf16.KernelRealResult
