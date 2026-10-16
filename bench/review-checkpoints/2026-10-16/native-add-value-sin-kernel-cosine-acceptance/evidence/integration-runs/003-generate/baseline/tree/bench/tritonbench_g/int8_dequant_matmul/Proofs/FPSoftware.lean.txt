import bench.tritonbench_g.int8_dequant_matmul.Proofs.FPComplete
namespace VeriTile.Bench.TritonBenchG.Int8DequantMatmul.FPSoftware
open VeriTile.Triton FP.Structural FPTyped
open Int4Matmul.FPSoftware (Word)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
abbrev scalar := Int4Matmul.FPSoftware.scalar
/-- The independent reference reads initial bytes, computes modulo-32 integer
dot/accumulation, converts the signed accumulator to fp32, performs each fp32
multiply, narrows once to fp16, and optionally performs a direct fp16 addition. -/
def machineOutput (p : Parameters) (s : State Word) (divisor : Word) (i : CIndex p) : Word :=
  let x := (s.mem p.rowScale (inputRow p s i.1)).read .fp32
  let w := (s.mem p.columnScale (inputColumn p s i.2.1)).read .fp32
  let signedAcc := FP.round .fp32 .ieeeValue (accumulation p s p.steps i : Rat)
  let half := QuantizeKvTransform.MixedBits.narrow .ieeeValue
    (FP.mul .ieeeValue w (FP.mul .ieeeValue x (FP.mul .ieeeValue signedAcc divisor)))
  if p.hasBias then QuantizeKvTransform.MixedBits.embed
    (MatmulDequantInt4.Binary16.add .ieeeValue (QuantizeKvTransform.MixedBits.half half)
      (QuantizeKvTransform.MixedBits.half
        ((s.mem p.bias (columnBlock p s * p.blockColumns + i.2.1.val)).read .fp16)))
  else half

theorem output_scalar (rest : Algebra Word) (p : Parameters) (s : State Word) (divisor : Word) :
    outputValue (scalar rest) p s divisor = machineOutput p s divisor := by
  funext i
  simp [outputValue, machineOutput, rowFactor, columnFactor,
    scalar, Int4Matmul.FPSoftware.scalar, FP.Software.scalarFP32, FP.Software.fp32Context,
    QuantizeKvTransform.MixedBits.half_embedding]

theorem correctness (rest : Algebra Word) (p : Parameters) (s : State Word) (divisor : Word)
    (hdiv : s.regs .fp32 [] "divfactor" = some (fun _ => divisor))
    (hw : ∀ i j, active p s i → active p s j → outputOffset p s i = outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (FPTyped.backend (scalar rest)) (source p) s = some final ∧
      Post p s final (machineOutput p s divisor) := by
  obtain ⟨final, he, hp⟩ := FPTyped.correctness (scalar rest) p s divisor hdiv hw
  exact ⟨final, he, output_scalar rest p s divisor ▸ hp⟩

#axiomsClean output_scalar
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.Int8DequantMatmul.FPSoftware
