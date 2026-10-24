import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPShape
import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPWitness
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4
open VeriTile.Triton FP.Structural
open Int4Matmul.FPSoftware (Word)
noncomputable section
/-! Both original NO_GROUPS branches and all K iterations. Binary16 products
and subtraction retain their separate source rounding points; D is an explicit
typed fp16×fp16→fp32 dot. Compiler/GPU dot and contraction binding is separate. -/
specification matmul_dequantize_int4_fp16_correctness
    (rest : Algebra Word) (D : FPTyped.Dot Word) (p : FPTyped.Parameters) (s : State Word)
    (hInputs : FPTyped.PackedInputs p s)
    (hw : ∀ i j, FPTyped.active p s i → FPTyped.active p s j →
      FPTyped.outputOffset p s i = FPTyped.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (FPTyped.backend (FPSoftware.scalar rest) D)
      (FPTyped.source p) s = some final ∧
      FPTyped.Post p s final (FPSoftware.machineOutput D p s) ∧
      ∀ step, step < FPTyped.steps p → ∀ i : FPTyped.BIndex p, ∃ word zero : BitVec 32,
        s.mem p.weight (FPTyped.bOffset p s step i) = .mk .int word.toInt ∧
        s.mem p.zero (FPTyped.zeroOffset p s step (i.2.1, PUnit.unit)) = .mk .int zero.toInt ∧
        FPTyped.weightNibble p s step i = ((word >>> (i.1.val % 8 * 4)) &&& (15 : BitVec 32)).toInt ∧
        FPTyped.zeroNibble p s step (i.2.1, PUnit.unit) =
          ((zero >>> (FPTyped.inputColumn p s i.2.1 % 8 * 4)) &&& (15 : BitVec 32)).toInt := by
  obtain ⟨final, he, hp⟩ := FPSoftware.correctness rest D p s hw
  exact ⟨final, he, hp, FPTyped.packed_nibbles_bridge p s hInputs⟩

theorem matmul_dequantize_int4_bit_success (rest : Algebra Word) (mode : Bool) :
    ∃ final, FP.Structural.exec (FPTyped.backend (FPSoftware.scalar rest) FPSoftware.softwareDot)
      (FPTyped.source (FPSoftware.witnessParameters mode)) FPSoftware.witnessInitial = some final ∧
      final.mem "output" 0 = .mk .fp16 (FPSoftware.halfResult mode) ∧
      final.mem "output" 16 = FPSoftware.witnessInitial.mem "output" 16 ∧
      final.mem "output" 255 = FPSoftware.witnessInitial.mem "output" 255 ∧
      final.mem "weight" 7 = .mk .int FPSoftware.weightWord.toInt ∧
      (-7 * (1365 / 4096) : Rat) ≠ -1195 / 512 := FPSoftware.bit_success rest mode

#axiomsClean matmul_dequantize_int4_fp16_correctness
#axiomsClean matmul_dequantize_int4_bit_success
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4
