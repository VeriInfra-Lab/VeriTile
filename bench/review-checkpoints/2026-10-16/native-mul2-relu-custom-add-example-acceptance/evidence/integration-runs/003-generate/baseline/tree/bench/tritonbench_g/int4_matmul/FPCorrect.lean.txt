import bench.tritonbench_g.int4_matmul.Proofs.FPShape
import bench.tritonbench_g.int4_matmul.Proofs.FPWitness
namespace VeriTile.Bench.TritonBenchG.Int4Matmul
open VeriTile.Triton FP.Structural
noncomputable section
/-! Full accepted SPLIT_K=1 K-loop, with actual int32 packed storage, direct
binary16 dequantization and binary32 accumulation. D is the explicit typed
half×half→float dot primitive; its GPU implementation binding is separate.
The independent softwareDot instance has a fixed fp32 reduction schedule. -/
specification int4_matmul_fp16_correctness
    (rest : Algebra FPSoftware.Word) (D : FPTyped.Dot FPSoftware.Word)
    (p : FPTyped.Parameters) (s : State FPSoftware.Word)
    (hTrip : p.inner = p.blockInner * p.steps) (hSplit : s.pids 1 = 0)
    (hInputs : FPTyped.PackedInputs p s)
    (hw : ∀ i j, FPTyped.active p s i → FPTyped.active p s j →
      FPTyped.outputOffset p s i = FPTyped.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (FPTyped.backend (FPSoftware.scalar rest) D)
      (FPTyped.source p) s = some final ∧
      FPTyped.Post p s final (FPSoftware.machineOutput D p s) ∧
      ∀ step, step < p.steps → ∀ i : FPTyped.BIndex p, ∃ word zero : BitVec 32,
        s.mem p.weight (FPTyped.bOffset p s step i) = .mk .int word.toInt ∧
        s.mem p.zero (FPTyped.zeroOffset p s step i) = .mk .int zero.toInt ∧
        FPTyped.difference p s step i = FPTyped.packedDifference p s i word zero := by
  obtain ⟨final, he, hp⟩ := FPSoftware.correctness rest D p s hw
  exact ⟨final, he, hp, FPTyped.packed_difference_bridge p s hInputs⟩

theorem int4_matmul_bit_success (rest : Algebra FPSoftware.Word) :
    ∃ final, FP.Structural.exec (FPTyped.backend (FPSoftware.scalar rest) FPSoftware.softwareDot)
      (FPTyped.source FPSoftware.witnessParameters) FPSoftware.witnessInitial = some final ∧
      final.mem "output" 0 = .mk .fp16 FPSoftware.halfResult ∧
      final.mem "output" 16 = FPSoftware.witnessInitial.mem "output" 16 ∧
      final.mem "weight" 7 = .mk .int FPSoftware.weightWord.toInt ∧
      (-7 * (1365 / 4096) : Rat) ≠ -597 / 256 := FPSoftware.bit_success rest

#axiomsClean int4_matmul_fp16_correctness
#axiomsClean int4_matmul_bit_success
end
end VeriTile.Bench.TritonBenchG.Int4Matmul
