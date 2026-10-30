import bench.repairs.tritonbench_g.swiglu_backward.Proofs.NativeIO
import bench.repairs.tritonbench_g.swiglu_backward.Proofs.IOImplements
import bench.tritonbench_g.swiglu_backward.Proofs.Observed

namespace VeriTile.Bench.Repairs.SwigluBackward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward
open Whole
noncomputable section

/-- Independent native recurrence, retaining casts, operand order, sigmoid
and contracted FMA from the pinned compiler graph. -/
def nativeFormula (M : Observed.NativeModel (BitVec 32)) (ch : Channel)
    (x y d : BitVec 32) : BitVec 32 :=
  let x := M.cast x
  let y := M.cast y
  let d := M.cast d
  let s := M.sigmoid x
  match ch with
  | .dx => M.mul (M.mul (M.mul s (M.fma (M.sub (M.literal 1) s) x (M.literal 1))) y) d
  | .dy => M.mul (M.mul s x) d
  | .out => M.mul (M.mul s x) y

/-- All FP32 input words yield the independent ordered native formulas for
both gradients and optional forward output. No native-to-Real identity or
floating transformation assumption is introduced. -/
theorem swiglu_backward_kernel_fp32 (M : Observed.NativeModel (BitVec 32))
    (c : Layout) (B : Nat) (R : Bool) (positive : 0 < B) (own : Ownership c R)
    (tiles : List (Nat × Nat)) (schedule : tiles.Perm (grid c B)) :
    (swigluNativeIO M c B R tiles).Implements (IO.reference c (nativeFormula M)) := by
  exact IOProof.implements Observed.graphAlgebra Observed.Term.input (Observed.interpret M)
    (Observed.reference M) (Observed.source_graph_lowering M) c B R positive own tiles schedule

#axiomsClean swiglu_backward_kernel_fp32
#specNonCircular nativeFormula avoiding [Observed.interpret,scalar,runTiles]
#stmtSurfaceSubset swiglu_backward_kernel_fp32 ⊆
  [Observed.NativeModel, Observed.Term, Layout, Ownership, grid, swigluNativeIO,
   ProgramIO.Implements, IO.reference, nativeFormula,
   Observed.instInhabitedTerm]
end
end VeriTile.Bench.Repairs.SwigluBackward
