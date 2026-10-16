import bench.tritonbench_g.int8_dequant_matmul.Proofs.FPStorage
import bench.tritonbench_g.int8_dequant_matmul.Proofs.FPWitness
namespace VeriTile.Bench.TritonBenchG.Int8DequantMatmul
open VeriTile.Triton FP.Structural
open Int4Matmul.FPSoftware (Word)
noncomputable section
/-! The original SPLIT_K=1, EVEN_K=true scope, every K iteration, both bias
branches. Inputs use signed byte words; accumulation wraps to int32 after each
block. All floating scalar operations use concrete software fp32/fp16 bits.
GPU/compiler binding of this interpreter remains a separate native obligation. -/
specification int8_dequant_matmul_fp_correctness
    (rest : Algebra Word) (p : FPTyped.Parameters) (s : State Word) (divisor : Word)
    (hK : p.inner = p.blockInner * p.steps) (hpid1 : s.pids 1 = 0)
    (hInputs : FPTyped.ByteInputs p s)
    (hdiv : s.regs .fp32 [] "divfactor" = some (fun _ => divisor))
    (hw : ∀ i j, FPTyped.active p s i → FPTyped.active p s j →
      FPTyped.outputOffset p s i = FPTyped.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (FPTyped.backend (FPSoftware.scalar rest))
      (FPTyped.source p) s = some final ∧
      FPTyped.Post p s final (FPSoftware.machineOutput p s divisor) ∧
      (∀ k, k < p.steps → ∀ i : FPTyped.AIndex p, ∃ w : BitVec 8,
        s.mem p.input (FPTyped.aOffset p s k i) = .mk .int w.toInt ∧
        FPTyped.aValue p s k i = w.toInt) ∧
      (∀ k, k < p.steps → ∀ i : FPTyped.BIndex p, ∃ w : BitVec 8,
        s.mem p.weight (FPTyped.bOffset p s k i) = .mk .int w.toInt ∧
        FPTyped.bValue p s k i = w.toInt) := by
  obtain ⟨final, he, hp⟩ := FPSoftware.correctness rest p s divisor hdiv hw
  exact ⟨final, he, hp, FPTyped.byte_inputs_bridge p s hInputs⟩

theorem int8_dequant_matmul_bit_success (rest : Algebra Word) (mode : Bool) :
    ∃ final, FP.Structural.exec (FPTyped.backend (FPSoftware.scalar rest))
      (FPTyped.source (FPSoftware.witnessParameters mode)) FPSoftware.witnessInitial = some final ∧
      final.mem "output" 0 = .mk .fp16 (FPSoftware.halfResult mode) ∧
      final.mem "output" 31 = FPSoftware.witnessInitial.mem "output" 31 ∧
      final.mem "output" 495 = FPSoftware.witnessInitial.mem "output" 495 ∧
      final.mem "weight" 7 = .mk .int FPSoftware.weightWord.toInt ∧
      ((-1040384 : Rat) / 16129) ≠ -129 / 2 := FPSoftware.bit_success rest mode

#axiomsClean int8_dequant_matmul_fp_correctness
#axiomsClean int8_dequant_matmul_bit_success
end
end VeriTile.Bench.TritonBenchG.Int8DequantMatmul
