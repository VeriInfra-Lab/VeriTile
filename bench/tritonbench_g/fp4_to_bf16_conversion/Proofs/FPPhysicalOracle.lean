import bench.tritonbench_g.fp4_to_bf16_conversion.FPCorrect
import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.FPFormatOracle

namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.PhysicalOracle
open VeriTile.Triton FP.Structural FPValue
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- The physical uint8 scale-load relation. The abstract interpreter's Nat
channel is unbounded; this relation is precisely storage typing, including
masked undef. It does not set a masked load to zero or require a complete
scale group. The older full arbitrary-Nat contract remains unchanged. -/
def ScaleBytes (p : Parameters) (initial : State Word) : Prop :=
  ∀ i : Fin p.groups, ∃ b : BitVec 8, loadedScale p initial i.val = b.toNat

theorem scale_group_in_bounds (p : Parameters) (j : TileIndex [p.block*2]) :
    j.1.val / 32 < p.groups := by
  have h := j.1.isLt
  change j.1.val < p.groups*16*2 at h
  omega

theorem reference_bits (p : Parameters) (initial : State Word)
    (j : TileIndex [p.block*2]) (b : BitVec 8)
    (hb : loadedScale p initial (j.1.val/32) = b.toNat) :
    (bf16Word (expectedValue software p initial j)).bits =
      FormatOracle.expectedBits (initialNibble p initial j) b := by
  simpa only [expectedValue, byteValue,
    VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue.expectedValue, hb] using
      FormatOracle.all_nibble_scale_pairs (initialNibble p initial j) b

/-- Complete candidate execution with the independently derived integer
format oracle at every active output, plus the original exact memory/frame
claims. Quantifiers retain every count/group/program and all aliasing. The
only added corollary premise is uint8 typing of physical scale/undef payloads;
there is no rounding law, finite premise, output equality or count%16 limit. -/
theorem physical_uint8_refinement
    (rest : Algebra Word) (p : Parameters) (initial : State Word)
    (hbytes : ActiveBytes p initial) (hscales : ScaleBytes p initial) :
    ∃ final, FP.Structural.exec (backend rest software) (source p) initial = some final ∧
      final.mem = (written software p initial).mem ∧
      (∀ j : TileIndex [p.block*2], active p initial j →
        ∃ b : BitVec 8, loadedScale p initial (j.1.val/32) = b.toNat ∧
          (bf16Word ((final.mem p.output (outputOffset p initial j)).read .bf16)).bits =
            FormatOracle.expectedBits (initialNibble p initial j) b) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ j : TileIndex [p.block*2],
        active p initial j → o ≠ outputOffset p initial j) → final.mem r o = initial.mem r o) := by
  obtain ⟨final, hexec, hmem, hread, hframe⟩ :=
    FPCorrect.fp4_to_bf16_scaled_candidate_machine_correctness rest p initial hbytes
  refine ⟨final, hexec, hmem, ?_, hframe⟩
  intro j hj
  obtain ⟨b, hb⟩ := hscales ⟨j.1.val/32, scale_group_in_bounds p j⟩
  refine ⟨b, hb, ?_⟩
  rw [hread j hj, Cell.read_same]
  exact reference_bits p initial j b hb

#specNonCircular ScaleBytes avoiding [source, FP.Structural.exec, written, expectedValue]
#axiomsClean scale_group_in_bounds
#axiomsClean reference_bits
#axiomsClean physical_uint8_refinement
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.PhysicalOracle
