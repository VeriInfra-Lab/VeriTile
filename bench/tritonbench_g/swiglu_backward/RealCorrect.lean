import bench.tritonbench_g.swiglu_backward.Proofs.LogicalIO
import bench.tritonbench_g.swiglu_backward.Proofs.Mathematical_WholePublic

namespace VeriTile.Bench.Repairs.SwigluBackward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Analytic input gradients and optional forward value. -/
def backwardFormula (ch : Channel) (x y d : ℝ) : ℝ :=
  let s := 1 / (1 + Real.exp (-x))
  match ch with
  | .dx => d*y*s*(1+x*(1-s))
  | .dy => d*x*s
  | .out => x*s*y

/-- Complete FP32 source grid interpreted over Real, with explicit strided
matrix windows and the recompute-dependent forward output mask. -/
abbrev swigluIO (c : Layout) (B : Nat) (R : Bool) (tiles : List (Nat × Nat)) :=
  Logical.io c B R tiles

/-- Independent analytic gradient and optional forward formulas on logical
matrix inputs. Access safety, execution, output binding and frame use shared IO. -/
specification swiglu_backward_kernel_correct (c : Layout) (B : Nat) (R : Bool)
    (positive : 0 < B) (own : Ownership c R) (tiles : List (Nat × Nat))
    (schedule : tiles.Perm (grid c B)) :
    VeriTile.Spec.Real (swigluIO c B R tiles ⊨ fun xs ch row col =>
      backwardFormula ch (xs.x row col) (xs.y row col) (xs.dout row col)) :=
  Logical.reader_implements c B R positive own tiles schedule

#axiomsClean swiglu_backward_kernel_correct
#specNonCircular backwardFormula avoiding [Layout.source,runTiles,scalar]
#stmtSurfaceSubset swiglu_backward_kernel_correct ⊆
  [VeriTile.Spec.Real, Layout,Layout.rows,Layout.cols,Channel,Ownership,grid,swigluIO,
   LogicalProgramIO.Implements,Logical.Inputs,backwardFormula,Logical.Inputs.x,Logical.Inputs.y,Logical.Inputs.dout]
end
end VeriTile.Bench.Repairs.SwigluBackward
