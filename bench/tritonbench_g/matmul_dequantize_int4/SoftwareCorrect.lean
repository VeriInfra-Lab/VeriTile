import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPSourceComplete
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4
open VeriTile.Triton FP.Structural
open Int4Matmul.FPSoftware (Word)
noncomputable section
/-! Complete original-source software profile. The original source graph uses
dot-from-positive-zero with separate fp32 multiply/add and then a separate
rounded add to the live accumulator. This entry does not translate it into
compiled dotAcc, does not contract half operations, and does not claim native
compiler/GPU equivalence. No arbitrary numerical callback remains. -/
specification matmul_dequantize_int4_original_source_bits_correctness
    (p : FPTyped.Parameters) (s : State Word)
    (hInputs : FPTyped.PackedInputs p s)
    (hw : ∀ i j, FPTyped.active p s i → FPTyped.active p s j →
      FPTyped.outputOffset p s i = FPTyped.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec
      (FPTyped.backend (FPSoftware.scalar SourceBits.unused) FPSoftware.softwareDot)
      (FPTyped.source p) s = some final ∧
      final.mem = (SourceBits.sourceReference p s).mem ∧
      FPTyped.Post p s final (SourceBits.sourceOutput p s) ∧
      ∀ step, step < FPTyped.steps p → ∀ i : FPTyped.BIndex p, ∃ word zero : BitVec 32,
        s.mem p.weight (FPTyped.bOffset p s step i) = .mk .int word.toInt ∧
        s.mem p.zero (FPTyped.zeroOffset p s step (i.2.1, PUnit.unit)) = .mk .int zero.toInt ∧
        FPTyped.weightNibble p s step i = ((word >>> (i.1.val % 8 * 4)) &&& (15 : BitVec 32)).toInt ∧
        FPTyped.zeroNibble p s step (i.2.1, PUnit.unit) =
          ((zero >>> (FPTyped.inputColumn p s i.2.1 % 8 * 4)) &&& (15 : BitVec 32)).toInt := by
  obtain ⟨final,he,hp⟩ := SourceBits.source_readback_correctness SourceBits.unused p s hw
  obtain ⟨other,ho,hm⟩ := SourceBits.source_memory_correctness SourceBits.unused p s
  have same : final=other := Option.some.inj (he.symm.trans ho)
  exact ⟨final,he,same.symm ▸ hm,hp,FPTyped.packed_nibbles_bridge p s hInputs⟩

/-- Existing full two-iteration, 16×16, partially masked, negative-packed-word
witness, now with no fallback parameter. Both NO_GROUPS branches read back
nonzero rounded outputs and preserve inactive output and source cells. -/
theorem matmul_dequantize_int4_original_source_bit_success (mode : Bool) :
    ∃ final, FP.Structural.exec
      (FPTyped.backend (FPSoftware.scalar SourceBits.unused) FPSoftware.softwareDot)
      (FPTyped.source (FPSoftware.witnessParameters mode)) FPSoftware.witnessInitial = some final ∧
      final.mem "output" 0 = .mk .fp16 (FPSoftware.halfResult mode) ∧
      final.mem "output" 16 = FPSoftware.witnessInitial.mem "output" 16 ∧
      final.mem "output" 255 = FPSoftware.witnessInitial.mem "output" 255 ∧
      final.mem "weight" 7 = .mk .int FPSoftware.weightWord.toInt ∧
      (-7 * (1365 / 4096) : Rat) ≠ -1195 / 512 :=
  FPSoftware.bit_success SourceBits.unused mode

#axiomsClean matmul_dequantize_int4_original_source_bits_correctness
#axiomsClean matmul_dequantize_int4_original_source_bit_success
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4
