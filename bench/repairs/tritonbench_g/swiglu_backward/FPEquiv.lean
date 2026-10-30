import bench.repairs.tritonbench_g.swiglu_backward.Proofs.NativeIO
import bench.repairs.tritonbench_g.swiglu_backward.Proofs.GridEquivalence
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.SwigluBackward
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward
open Whole
open scoped VeriTile.Spec
noncomputable section

/-- Complete selected original source grid and its existing native observer. -/
def original (M : Observed.NativeModel (BitVec 32)) (c : Layout)
    (B : Nat) (R : Bool) (tiles : List (Nat × Nat)) :=
  ProgramIO.Implementation.original (swigluNativeIO M c B R tiles) (fun _ => True)

/-- Existing source candidate commutes only the natural row-address products. -/
def indexCommuted (M : Observed.NativeModel (BitVec 32)) (c : Layout)
    (B : Nat) (R : Bool) (tiles : List (Nat × Nat)) :=
  ProgramIO.Implementation.ofVariant
    (IO.commuted Observed.graphAlgebra Observed.Term.input (Observed.interpret M) c B R tiles)
    (fun _ => True)

def assumptions (M : Observed.NativeModel (BitVec 32)) (c : Layout)
    (B : Nat) (R : Bool) (tiles : List (Nat × Nat)) :
    Spec.Assumptions (ProgramIO.Implementation (swigluNativeIO M c B R tiles) (fun _ => True)) := []

/-- Actual complete FP32 source programs, with both recomputation branches,
every legal full-grid order and every input word. The integer address change
preserves casts, sigmoid, FMA/operand order, all three stores and native observers.
Safety, successful runs, output equality and full exterior frames are proved. -/
specification swiglu_backward_source_equiv (M : Observed.NativeModel (BitVec 32))
    (c : Layout) (B : Nat) (R : Bool) (positive : 0<B) (own : Ownership c R)
    (tiles : List (Nat × Nat)) (schedule : tiles.Perm (grid c B)) :
    original M c B R tiles ≡[assumptions M c B R tiles] indexCommuted M c B R tiles :=
  ProgramIO.Implementation.of_equivalent _
    (GridEquivalence.equivalent Observed.graphAlgebra Observed.Term.input
      (Observed.interpret M) c B R positive own tiles schedule)

#print_fp_assumptions swiglu_backward_source_equiv
#axiomsClean swiglu_backward_source_equiv
end
end VeriTile.Bench.Repairs.SwigluBackward
