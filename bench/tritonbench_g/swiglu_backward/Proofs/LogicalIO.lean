import bench.tritonbench_g.swiglu_backward.IO
import bench.tritonbench_g.swiglu_backward.Proofs.IOImplements
namespace VeriTile.Bench.Repairs.SwigluBackward.Logical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
noncomputable section
theorem judgment_iff (c : Layout) (B : Nat) (R : Bool) (tiles : List (Nat × Nat))
    (f : Channel → ℝ → ℝ → ℝ → ℝ) :
    (io c B R tiles).Implements (fun xs ch r col => f ch (xs.x r col) (xs.y r col) (xs.dout r col)) ↔
    (IO.swigluIO realAlgebra id id c B R tiles).Implements (IO.reference c f) := by
  have same : (io c B R tiles).lift (fun xs ch r col => f ch (xs.x r col) (xs.y r col) (xs.dout r col)) =
      IO.reference c f := by
    funext xs port
    fin_cases port <;> rfl
  change (IO.swigluIO realAlgebra id id c B R tiles).Implements _ ↔ _
  rw [same]
private def readerScalar (ch : Channel) (x y d : ℝ) : ℝ :=
  let s := 1 / (1 + Real.exp (-x))
  match ch with
  | .dx => d*y*s*(1+x*(1-s))
  | .dy => d*x*s
  | .out => x*s*y

theorem reader_implements (c : Layout) (B : Nat) (R : Bool)
    (positive : 0 < B) (own : Ownership c R) (tiles : List (Nat × Nat))
    (schedule : tiles.Perm (grid c B)) :
    (io c B R tiles).Implements (fun xs ch row col =>
      readerScalar ch (xs.x row col) (xs.y row col) (xs.dout row col)) := by
  apply (judgment_iff c B R tiles readerScalar).2
  apply IOProof.implements realAlgebra id id readerScalar _ c B R positive own tiles schedule
  intro ch x y d
  cases ch <;> simpa only [id_eq,readerScalar,formula,
    VeriTile.Bench.TritonBenchG.SwigluBackward.Math.dx,
    VeriTile.Bench.TritonBenchG.SwigluBackward.Math.dy,
    VeriTile.Bench.TritonBenchG.SwigluBackward.Math.forward,
    VeriTile.Bench.TritonBenchG.SwigluBackward.Math.sigmoid] using scalar_real _ x y d

end
end VeriTile.Bench.Repairs.SwigluBackward.Logical
