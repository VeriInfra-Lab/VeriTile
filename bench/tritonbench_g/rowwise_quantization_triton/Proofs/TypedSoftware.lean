import bench.tritonbench_g.rowwise_quantization_triton.Proofs.TypedScaled
import bench.tritonbench_g.quantize_copy_kv.Proofs.FPMaxProfile
namespace VeriTile.Bench.TritonBenchG.RowwiseQuantizationTriton.TypedScaled
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
abbrev Word := FP.Value .fp32
variable [Inhabited Word]

/-- Explicit fp32 scalar operations and default maxnum reduction, followed by
one direct binary32-to-binary16 rounding at the maximum store. -/
def bitBackend (rest : Algebra Word) (cfg : FP.Config) : Algebra Word :=
  QuantizeCopyKv.FPProfile.backend rest cfg (fun _ _ => 0)

def bitMaximum (cfg : FP.Config) (p : Parameters) (s : State Word) : Word :=
  QuantizeCopyKv.FPProfile.maximum cfg (fun i : Fin p.width =>
    if i.val < p.columns then ((s.mem p.x (s.pids 0 * p.columns + i.val)).read .real).abs
    else FP.Value.zero .fp32)

def bitValue (cfg : FP.Config) (p : Parameters) (s : State Word) (i : TileIndex [p.width]) : Word :=
  FP.mul cfg (FP.Value.ofBits .fp32 0x42fe0000)
    (FP.div cfg ((s.mem p.x (offset p s i)).read .real) (bitMaximum cfg p s))

def bitMaximumStored (cfg : FP.Config) (p : Parameters) (s : State Word) : Word :=
  QuantizeKvTransform.MixedBits.narrow cfg (bitMaximum cfg p s)

theorem maximum_bits (rest : Algebra Word) (cfg : FP.Config) (p : Parameters) (s : State Word) :
    maximum (bitBackend rest cfg) p s = bitMaximum cfg p s := by
  simp [maximum, maxInput, bitBackend, bitMaximum, offset,
    QuantizeCopyKv.FPProfile.backend, QuantizeKvTransform.MixedBits.backend, QuantizeKvCopy.TypedQuantize.backend,
    FP.IntegerConversion.fp32TotalBackend, FP.Software.scalarFP32, FP.Software.fp32Context,
    QuantizeCopyKv.FPProfile.reduceMaximum, TileShape.insertAxisIndex]
  rfl

theorem value_bits (rest : Algebra Word) (cfg : FP.Config) (p : Parameters) (s : State Word)
    (i : TileIndex [p.width]) : scaledValue (bitBackend rest cfg) p s i = bitValue cfg p s i := by
  unfold scaledValue
  rw [maximum_bits]
  simp [bitBackend, bitValue, QuantizeCopyKv.FPProfile.backend, QuantizeKvTransform.MixedBits.backend,
    QuantizeKvCopy.TypedQuantize.backend, FP.IntegerConversion.fp32TotalBackend,
    FP.Software.scalarFP32, FP.Software.fp32Context]

theorem maximum_stored_bits (rest : Algebra Word) (cfg : FP.Config) (p : Parameters) (s : State Word) :
    maximumStored (bitBackend rest cfg) p s = bitMaximumStored cfg p s := by
  unfold maximumStored
  rw [maximum_bits]
  rfl

#axiomsClean maximum_bits
#axiomsClean value_bits
#axiomsClean maximum_stored_bits
end
end VeriTile.Bench.TritonBenchG.RowwiseQuantizationTriton.TypedScaled
