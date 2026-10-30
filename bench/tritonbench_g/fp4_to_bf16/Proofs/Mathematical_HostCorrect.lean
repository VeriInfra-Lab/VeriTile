import bench.tritonbench_g.fp4_to_bf16.Proofs.Mathematical_Host

/-! Complete logical host-grid contract for the original unscaled FP4 wrapper.
The output is freshly allocated bf16 storage with doubled final dimension.
The launch executes the actual admitted kernel, in program-ID order, for the
original ceilDiv(numel,512) grid. This is not a native allocator/concurrency or
arbitrary-storage Python-wrapper refinement theorem. -/
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.HostCorrect
open VeriTile.Triton FP.Structural FPValue Host
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Read the bf16 payload from the task's tagged structural cell. -/
def readBf16 (c : Cell Word) : FP.Value .bf16 :=
  ⟨(c.read .bf16).bits.truncate 16⟩

private theorem read_expected (h : Input) (initial : State Word) (j : Nat) :
    readBf16 (cell h initial j) = ⟨Bits.expectedBits (nibble h initial j)⟩ := by
  simp only [readBf16, cell, value, embed, Cell.read_same, FP.Value.mk.injEq]
  simpa only [BitVec.setWidth_eq] using
    (BitVec.setWidth_setWidth_of_le (Bits.expectedBits (nibble h initial j)) (by decide : 16 ≤ 32))

/-- Every output of every original BLOCK=512 program has its exact independent
initial-byte value, including tails and both zero signs. The entire memory
frame is retained. PackedInput describes uint8 storage, and fresh is supplied
by the wrapper's new output allocation; no numerical law is assumed. The
shape may have any number of leading dimensions and any element count. -/
specification fp4_to_bf16_host_grid_correctness
    (rest : Algebra Word) (h : Input) (initial : State Word)
    (fresh : h.input ≠ h.output) (typed : PackedInput h initial) :
    ∃ final, Host.run rest h initial = some final ∧
      final.mem = expectedMemory h initial ∧
      h.shape.output.prod = h.shape.input.prod * 2 ∧
      (∀ j, j < h.shape.count * 2 →
        final.mem h.output j = cell h initial j ∧
        (readBf16 (final.mem h.output j)).decode =
          .finite (decide (8 ≤ (nibble h initial j).toNat))
            (Bits.magnitude (nibble h initial j))) ∧
      (∀ r o, (r ≠ h.output ∨ h.shape.count * 2 ≤ o) →
        final.mem r o = initial.mem r o) := by
  refine ⟨prefixState rest h initial (grid h), run_prefix rest h initial (grid h),
    launch_memory rest h initial fresh typed, ?_, ?_, ?_⟩
  · rw [input_numel, output_numel]
  · intro j hj
    have memory := launch_memory rest h initial fresh typed
    have output : (prefixState rest h initial (grid h)).mem h.output j = cell h initial j := by
      rw [memory]
      simp only [expectedMemory, hj, and_self, ↓reduceIte]
    refine ⟨output, ?_⟩
    rw [output, read_expected]
    exact expected_decode h initial j
  · intro r o outside
    rw [launch_memory rest h initial fresh typed]
    have inactive : ¬ (r = h.output ∧ o < h.shape.count * 2) := by
      rcases outside with hr | ho
      · exact fun hi => hr hi.1
      · omega
    simp only [expectedMemory, inactive, ↓reduceIte]

#axiomsClean fp4_to_bf16_host_grid_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.HostCorrect
