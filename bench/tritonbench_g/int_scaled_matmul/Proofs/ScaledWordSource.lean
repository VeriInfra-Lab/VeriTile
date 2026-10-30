import bench.tritonbench_g.int_scaled_matmul.Sources.Kernels
import bench.tritonbench_g.int_scaled_matmul.Sources.Mathematical_IntScaledMatmul
import VeriTile.Triton.Float.SignedCast
import VeriTile.Triton.Float.IntegerDot
import VeriTile.Triton.Float.ScalarBackend
import VeriTile.Triton.Float.IntegerConversionTotal
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ScaledWord
open VeriTile.Bench.TritonBenchG.IntScaledMatmul
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedVariables false
def wrap (w : Nat) (x : Int) : Int := (BitVec.ofInt w x).toInt

abbrev Word := FP.Value .fp32
local instance scaledSourceDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩
/-- fp32 scalar operations and width-checked signed conversion, with an
arbitrary fallback only where the source integer conversion is undefined. -/
def backend (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int) : Algebra Word :=
  let scalar := FP.Software.scalarFP32 rest cfg
  let converted := FP.IntegerConversion.fp32TotalBackend scalar u (fun _ _ => 0)
  { converted with
    fromInt := fun p n => if p=some .fp32 then FP.round .fp32 cfg (n : Rat) else converted.fromInt p n
    integerDot := fun _ {batch} {_} {_} {_} a b =>
      some (fun i => (FP.IntegerDot.wordDot batch a b i).toInt) }

theorem original_decomposition (p : Parameters) : (original p).surfaceBody =
    prelude p ++ [.forRange "j" 0 p.steps 1 (originalBody p)] ++ epilogue p := by rfl
#axiomsClean original_decomposition
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ScaledWord
