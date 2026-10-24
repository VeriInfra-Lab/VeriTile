import bench.tritonbench_g.matmul_dequantize.Proofs.FPDequant
import bench.tritonbench_g.matmul_dequantize.Proofs.FPFirstShape
import bench.tritonbench_g.matmul_dequantize.Proofs.FPFirstWitness
import bench.tritonbench_g.matmul_dequantize.Proofs.FPSecondShape
import bench.tritonbench_g.matmul_dequantize.Proofs.FPSecondWitness
namespace VeriTile.Bench.TritonBenchG.MatmulDequantize
open VeriTile.Triton FP.Structural
open Int4Matmul.FPSoftware (Word)
noncomputable section
/-! All three original accepted stages have independent initial-state contracts.
Both matmul stages cover their full K loops; the first covers both NO_GROUPS
branches. Native dot and scalar contraction binding remains explicit. -/
specification matmul_dequantize_standalone_fp16_correctness [Inhabited MatmulDequantInt4.Binary16.Value] (rest : Algebra MatmulDequantInt4.Binary16.Value) (p : FPTyped.Standalone.Parameters) (s : State MatmulDequantInt4.Binary16.Value)
    (hInputs : MatmulDequantInt4.FPTyped.PackedInputs p s)
    (hw : ∀ i j, MatmulDequantInt4.FPTyped.active p s i → MatmulDequantInt4.FPTyped.active p s j →
      MatmulDequantInt4.FPTyped.outputOffset p s i = MatmulDequantInt4.FPTyped.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (MatmulDequantInt4.FPTyped.software rest) (FPTyped.Standalone.source p) s = some final ∧
      MatmulDequantInt4.FPTyped.Post p s final (MatmulDequantInt4.FPTyped.roundedValue p s) ∧
      ∀ i, MatmulDequantInt4.FPTyped.active p s i → ∃ word zero : BitVec 32,
        s.mem p.weight (MatmulDequantInt4.FPTyped.weightOffset p s i) = .mk .int word.toInt ∧
        s.mem p.zero (MatmulDequantInt4.FPTyped.zeroOffset p s i) = .mk .int zero.toInt ∧
        final.mem p.output (MatmulDequantInt4.FPTyped.outputOffset p s i) =
          .mk .fp16 (MatmulDequantInt4.FPTyped.packedResult p s i word zero) :=
  FPTyped.Standalone.correctness rest p s hInputs hw

theorem matmul_dequantize_standalone_bit_success [Inhabited MatmulDequantInt4.Binary16.Value] (rest : Algebra MatmulDequantInt4.Binary16.Value) :
    ∃ final, FP.Structural.exec (MatmulDequantInt4.FPTyped.software rest)
      (FPTyped.Standalone.source MatmulDequantInt4.FPTyped.witnessParameters) MatmulDequantInt4.FPTyped.witnessInitial = some final ∧
      final.mem "output" 119 = .mk .fp16 MatmulDequantInt4.FPTyped.resultValue ∧
      final.mem "output" 120 = MatmulDequantInt4.FPTyped.witnessInitial.mem "output" 120 ∧
      final.mem "weight" 7 = .mk .int MatmulDequantInt4.FPTyped.weightWord.toInt ∧
      (-7 * (1365 / 4096) : Rat) ≠ -597 / 256 :=
  FPTyped.Standalone.bit_success rest

#axiomsClean matmul_dequantize_standalone_fp16_correctness
#axiomsClean matmul_dequantize_standalone_bit_success
specification matmul_dequantize_second_fp16_correctness
    (rest : Algebra Int4Matmul.FPSoftware.Word) (D : FPTypedSecond.Dot Int4Matmul.FPSoftware.Word)
    (p : FPTypedSecond.Parameters) (s : State Int4Matmul.FPSoftware.Word)
    (hTrip : p.inner = p.blockInner * p.steps) (hSplit : s.pids 1 = 0)
    (hInputs : FPTypedSecond.PackedInputs p s)
    (hw : ∀ i j, FPTypedSecond.active p s i → FPTypedSecond.active p s j →
      FPTypedSecond.outputOffset p s i = FPTypedSecond.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (FPTypedSecond.backend (Int4Matmul.FPSoftware.scalar rest) D)
      (FPTypedSecond.source p) s = some final ∧
      FPTypedSecond.Post p s final (FPSecondSoftware.machineOutput D p s) ∧
      ∀ step, step < p.steps → ∀ i : FPTypedSecond.BIndex p, ∃ word zero : BitVec 32,
        s.mem p.weight (FPTypedSecond.bOffset p s step i) = .mk .int word.toInt ∧
        s.mem p.zero (FPTypedSecond.zeroOffset p s step i) = .mk .int zero.toInt ∧
        FPTypedSecond.difference p s step i = FPTypedSecond.packedDifference p s i word zero := by
  obtain ⟨final, he, hp⟩ := FPSecondSoftware.correctness rest D p s hw
  exact ⟨final, he, hp, FPTypedSecond.packed_difference_bridge p s hInputs⟩

theorem matmul_dequantize_second_bit_success (rest : Algebra Int4Matmul.FPSoftware.Word) :
    ∃ final, FP.Structural.exec (FPTypedSecond.backend (Int4Matmul.FPSoftware.scalar rest) FPSecondSoftware.softwareDot)
      (FPTypedSecond.source FPSecondSoftware.witnessParameters) FPSecondSoftware.witnessInitial = some final ∧
      final.mem "output" 0 = .mk .fp16 FPSecondSoftware.halfResult ∧
      final.mem "output" 16 = FPSecondSoftware.witnessInitial.mem "output" 16 ∧
      final.mem "weight" 7 = .mk .int FPSecondSoftware.weightWord.toInt ∧
      (-7 * (1365 / 4096) : Rat) ≠ -597 / 256 := FPSecondSoftware.bit_success rest

#axiomsClean matmul_dequantize_second_fp16_correctness
#axiomsClean matmul_dequantize_second_bit_success
specification matmul_dequantize_first_fp16_correctness
    (rest : Algebra Word) (D : FPTypedFirst.Dot Word) (p : FPTypedFirst.Parameters) (s : State Word)
    (hInputs : FPTypedFirst.PackedInputs p s)
    (hw : ∀ i j, FPTypedFirst.active p s i → FPTypedFirst.active p s j →
      FPTypedFirst.outputOffset p s i = FPTypedFirst.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (FPTypedFirst.backend (FPFirstSoftware.scalar rest) D)
      (FPTypedFirst.source p) s = some final ∧
      FPTypedFirst.Post p s final (FPFirstSoftware.machineOutput D p s) ∧
      ∀ step, step < FPTypedFirst.steps p → ∀ i : FPTypedFirst.BIndex p, ∃ word zero : BitVec 32,
        s.mem p.weight (FPTypedFirst.bOffset p s step i) = .mk .int word.toInt ∧
        s.mem p.zero (FPTypedFirst.zeroOffset p s step (i.2.1, PUnit.unit)) = .mk .int zero.toInt ∧
        FPTypedFirst.weightNibble p s step i = ((word >>> (i.1.val % 8 * 4)) &&& (15 : BitVec 32)).toInt ∧
        FPTypedFirst.zeroNibble p s step (i.2.1, PUnit.unit) =
          ((zero >>> (FPTypedFirst.inputColumn p s i.2.1 % 8 * 4)) &&& (15 : BitVec 32)).toInt := by
  obtain ⟨final, he, hp⟩ := FPFirstSoftware.correctness rest D p s hw
  exact ⟨final, he, hp, FPTypedFirst.packed_nibbles_bridge p s hInputs⟩

theorem matmul_dequantize_first_bit_success (rest : Algebra Word) (mode : Bool) :
    ∃ final, FP.Structural.exec (FPTypedFirst.backend (FPFirstSoftware.scalar rest) FPFirstSoftware.softwareDot)
      (FPTypedFirst.source (FPFirstSoftware.witnessParameters mode)) FPFirstSoftware.witnessInitial = some final ∧
      final.mem "output" 0 = .mk .fp16 (FPFirstSoftware.halfResult mode) ∧
      final.mem "output" 16 = FPFirstSoftware.witnessInitial.mem "output" 16 ∧
      final.mem "output" 255 = FPFirstSoftware.witnessInitial.mem "output" 255 ∧
      final.mem "weight" 7 = .mk .int FPFirstSoftware.weightWord.toInt ∧
      (-7 * (1365 / 4096) : Rat) ≠ -1195 / 512 := FPFirstSoftware.bit_success rest mode

#axiomsClean matmul_dequantize_first_fp16_correctness
#axiomsClean matmul_dequantize_first_bit_success
end
end VeriTile.Bench.TritonBenchG.MatmulDequantize
