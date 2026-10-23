import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.RawKernel
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.Fp4ToBf16.KernelNativeResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host HostCorrect RealSpec
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Complete raw finite source execution and exact BF16 table bits, including
both zero signs, with all exterior tagged cells preserved. -/
theorem fp4_to_bf16_native (rest : Algebra Word) (h : Input) (storage : Finite.Storage h.shape.count)
    (jobs : List Nat) (initial : State Word) (fresh : h.input≠h.output)
    (typed : PackedInput h initial) (gridSafe : grid h≤2147483647)
    (order : jobs.Perm (List.range (grid h))) :
    ∃ final, Launch.run rest h jobs initial=some final ∧
      final.mem=Host.expectedMemory h initial ∧
      (∀ j,j<h.shape.count*2 →
        (readBf16 (final.mem h.output j)).bits=Bits.expectedBits (nibble h initial j)) ∧
      (∀ r o,r≠h.output ∨ h.shape.count*2≤o → final.mem r o=initial.mem r o) := by
  obtain ⟨final,run,memory,frame⟩ := RawKernel.complete rest h storage jobs initial fresh typed gridSafe order
  refine ⟨final,run,memory,?_,frame⟩
  intro j hj
  rw [memory]
  simp only [expectedMemory,hj,and_self,if_true]
  rw [read_cell]

 theorem native_masks (count pid li lo : Nat) (capacity : 4*count<9223372036854775808)
    (hp : pid<(count+511)/512) (hi : li<512) (ho : lo<1024) :
    Kernels.inputMask count pid li=decide (pid*512+li<count) ∧
    Kernels.outputMask count pid lo=decide (pid*1024+lo<2*count) :=
  Finite.masks_exact count pid li lo capacity hp hi ho

#axiomsClean fp4_to_bf16_native
#axiomsClean native_masks
#specNonCircular Bits.expectedBits avoiding [source,Kernels.source,Launch.run]
end
end VeriTile.Bench.Repairs.Fp4ToBf16.KernelNativeResult
