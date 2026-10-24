import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.KernelState

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Formula
open KernelResult
noncomputable section

/-- Exact signed-integer copy specification, independent of source execution. -/
def integer (rows cols : Nat) : Nat → Int :=
  target 0 (rows*cols) cols (fun i => signedValue (arangeWord i)) (fun _ => 0)

theorem integer_word (rows cols : Nat) :
    integer rows cols = fun i => signedValue (wordTarget rows cols i) := by
  have h := map_target signedValue 0 (rows*cols) cols arangeWord (fun _ => 0)
  simpa only [integer,wordTarget,signedValue,BitVec.toInt_zero] using h.symm

theorem integer_real (rows cols i : Nat) :
    (integer rows cols i : ℝ) = realTarget rows cols i := by
  rw [integer_word]
  exact int32_real_exact rows cols i

#axiomsClean integer_word
#axiomsClean integer_real
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.Formula
