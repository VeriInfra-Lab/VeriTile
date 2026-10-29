import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.TypedFormula
import VeriTile.Triton.Random.Proofs.CanonicalUniformBridge

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.ActualMathematics
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
open scoped BigOperators Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

def realSum {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values ℝ .real shape) : Values ℝ .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => ∑ j : Fin (TileShape.axisDim shape ax),
      xs (TileShape.insertAxisIndex shape ax out j)
  · exact fun out => ∑ j : Fin (TileShape.axisDim shape ax),
      xs (TileShape.replaceAxisIndex shape ax out j)

/-- Exact mathematical interpretation of the preserved source primitives.
The source still contains its precision markers and casts; this interpretation
is used only by the Real specification, never by the opaque FP equivalence. -/
def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y | _ => 0
  unary := fun _ op x => match op with | .sqrt => Real.sqrt x | _ => 0
  compareLt := fun _ _ => some (fun x y => decide (x < y))
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun word => (((FP.Value.mk word.bits : FP.Value .fp32).toRat?).getD 0 : ℝ)
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => 0
  reduceSum := fun _ {_} ax keep xs => realSum ax keep xs

def comparisons : TypedSource.Comparisons algebra where
  lt := fun _ _ x y => decide (x < y)
  available := by intros; rfl

end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.ActualMathematics
