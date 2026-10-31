import bench.tritonbench_g.triton_argmax.Proofs.TypedScalarBinding

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedIntegerMacro
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
open VeriTile.Bench.TritonBenchG.MatrixReduction
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def backend (kind : Formats.IntegerKind) (p : ScalarBinding.IntegerPrimitives kind)
    (rest : Algebra Word) : Algebra Word :=
  { Typed.backend rest with
    floatToInt := fun _ _ _ => some (fun word => some ((ConversionMacro.SaturatingConversion.mk kind).native p word)) }

def written (kind : Formats.IntegerKind) (p : ScalarBinding.IntegerPrimitives kind)
    (s : State Word) (X V : RegionName) (I : Region .int)
    (size block base : Nat) (positive : 0<block) : State Word :=
  let selected := Typed.firstIndex s X size block base positive
  (s.write V (s.pids 0+base) (.mk .int
    ((ConversionMacro.SaturatingConversion.mk kind).native p ((s.mem X selected).read .real)))).write
    (Region.cast I) (s.pids 0+base) (.mk .int (Int.ofNat selected))

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedIntegerMacro
