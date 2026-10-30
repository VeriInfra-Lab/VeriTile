import bench.tritonbench_g.matmul_dequant_int4.Sources.Mathematical_FPWitness
import bench.tritonbench_g.matmul_dequant_int4.Sources.Mathematical_FPShape
namespace VeriTile.Bench.TritonBenchG.MatmulDequantInt4
open VeriTile.Triton FP.Structural
noncomputable section

/-- The original standalone dequantize stage, with genuine signed int32 input
words, fixed-width nibble semantics and concrete binary16 conversion/multiply.
The active address injectivity and input representation are ordinary input
conditions. The independent packed-bit result is derived from initial memory. -/
specification matmul_dequant_int4_fp16_correctness [Inhabited Binary16.Value]
    (rest : Algebra Binary16.Value) (p : FPTyped.Parameters) (s : State Binary16.Value)
    (hInputs : FPTyped.PackedInputs p s)
    (hw : ∀ i j, FPTyped.active p s i → FPTyped.active p s j →
      FPTyped.outputOffset p s i = FPTyped.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (FPTyped.software rest) (FPTyped.source p) s = some final ∧
      FPTyped.Post p s final (FPTyped.roundedValue p s) ∧
      ∀ i, FPTyped.active p s i → ∃ word zero : BitVec 32,
        s.mem p.weight (FPTyped.weightOffset p s i) = .mk .int word.toInt ∧
        s.mem p.zero (FPTyped.zeroOffset p s i) = .mk .int zero.toInt ∧
        final.mem p.output (FPTyped.outputOffset p s i) = .mk .fp16 (FPTyped.packedResult p s i word zero) := by
  obtain ⟨final, he, hp⟩ := FPTyped.software_correctness rest p s hw
  refine ⟨final, he, hp, ?_⟩
  intro i hi
  obtain ⟨w, z, hw, hz, hv⟩ := FPTyped.packed_result_bridge p s hInputs i hi
  exact ⟨w, z, hw, hz, hv ▸ hp.1 i hi⟩

/-- The bridge from actual signed int32 source shifts to the existing packed
unsigned-nibble interpretation; valid for every 32-bit word, including the
negative half of the encoding space. -/
theorem matmul_dequant_int4_signed_decode (word : BitVec 32) (slot : Fin 8) :
    (word.sshiftRight (slot.val * 4) &&& (15 : BitVec 32)) =
      ((word >>> (slot.val * 4)) &&& (15 : BitVec 32)) :=
  FPBits.signed_logical_mask word slot

theorem matmul_dequant_int4_bit_success [Inhabited Binary16.Value] (rest : Algebra Binary16.Value) :
    ∃ final, FP.Structural.exec (FPTyped.software rest) (FPTyped.source FPTyped.witnessParameters)
      FPTyped.witnessInitial = some final ∧
      final.mem "output" 119 = .mk .fp16 FPTyped.resultValue ∧
      final.mem "output" 120 = FPTyped.witnessInitial.mem "output" 120 ∧
      final.mem "weight" 7 = .mk .int FPTyped.weightWord.toInt ∧
      (-7 * (1365 / 4096) : Rat) ≠ -597 / 256 :=
  FPTyped.nonempty_software rest

theorem matmul_dequant_int4_source_signed_shape (p : FPTyped.Parameters) :
    (FPTyped.source p).surfaceBody[16]? = some (.assign .fp16 [p.blockRows, p.blockColumns]
      "fp_weight" (.alg (.mul .fp16 Broadcast.nil.consSame.consSame
        (.withPrecision .fp16 .fp32 (.castFloat .real .fp16 (.intToReal (.sub .int Broadcast.nil.consSame.consSame
          (Op.intBitAnd 32 .scalarR
            (Op.intShiftRight Broadcast.nil.consR.consSame
              (.ref .int [p.blockRows, p.blockColumns] "int32_b")
              (.ref .nat [p.blockRows, 1] "b_shift")) (.constInt 15))
          (Op.intBitAnd 32 .scalarR
            (Op.intShiftRight Broadcast.nil.consSame.consR
              (.ref .int [p.blockRows, p.blockColumns] "zp_b")
              (.ref .nat [1, p.blockColumns] "bzp_shift")) (.constInt 15))))))
        (.withPrecision .fp16 .fp32 (.ref .fp16 [p.blockRows, p.blockColumns] "scale_b"))))) := FPTyped.source_signed_shape p

#axiomsClean matmul_dequant_int4_source_signed_shape
#axiomsClean matmul_dequant_int4_fp16_correctness
#axiomsClean matmul_dequant_int4_signed_decode
#axiomsClean matmul_dequant_int4_bit_success
end
end VeriTile.Bench.TritonBenchG.MatmulDequantInt4
