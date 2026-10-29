import bench.tritonbench_g.fp4_to_bf16.Kernels
import bench.tritonbench_g.fp4_to_bf16.Proofs.Real
import bench.tritonbench_g.fp4_to_bf16.Proofs.IndexOverflow

/-! Public mathematical entry. This preserves the full historical logical
kernel/grid/order contracts and adds the independent e2m1 Real formula, exact
BF16 bits and frame. `IndexOverflow.permitted_allocation_counterexample`
refutes extension to the original unrestricted finite-index wrapper: its i32
output count overflows at n=2^30. No original Python has been repaired and no
native whole-wrapper correctness is claimed. -/
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.Correct
open VeriTile.Triton FP.Structural FPValue Host HostCorrect HostSchedule RealSpec
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Mathematical correctness of the actual admitted logical source, for all
nonscalar shapes and any complete program order. Packed uint8 is explicit;
no real-arithmetic approximation or FP numerical assumption is used. -/
specification fp4_to_bf16_real_correctness
    (rest : Algebra Word) (h : Input) (jobs : List Nat) (initial : State Word)
    (fresh : h.input ≠ h.output) (typed : PackedInput h initial)
    (order : jobs.Perm (List.range (grid h))) :
    ∃ final, runSchedule rest h jobs initial = some final ∧
      (∀ j, j < h.shape.count * 2 →
        (readBf16 (final.mem h.output j)).bits = Bits.expectedBits (nibble h initial j) ∧
        finiteReal (readBf16 (final.mem h.output j)) = some (e2m1 (nibble h initial j))) ∧
      (∀ r o, (r ≠ h.output ∨ h.shape.count * 2 ≤ o) →
        final.mem r o = initial.mem r o) :=
  schedule_real_correctness rest h jobs initial fresh typed order

#axiomsClean fp4_to_bf16_real_correctness
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.Correct
