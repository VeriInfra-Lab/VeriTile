import bench.tritonbench_g.nested_loops_processing.Proofs.TypedMemory
import bench.tritonbench_g.nested_loops_processing.Proofs.Real
import bench.tritonbench_g.nested_loops_processing.Proofs.Schedule

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.KernelResult
open VeriTile.Triton FP.Structural TypedExecution
noncomputable section
variable {α : Type} [Inhabited α]

/-- Only the native launch branch is modeled; empty buffers skip it before
pointer queries, exactly as in the Python wrapper. -/
def launch (M : Algebra α) (candidate : Bool) (count cols : Nat) (s : State α) : Option (State α) :=
  if count=0 then some s else exec M (Kernels.kernel candidate "input" "output" count cols) s

theorem empty_launch (M : Algebra α) (candidate : Bool) (cols : Nat) (s : State α) :
    launch M candidate 0 cols s = some s := by simp [launch]

def wordTarget (rows cols : Nat) := target (0 : Word) (rows*cols) cols arangeWord (fun _ ↦ 0)

theorem initialized_cell (s : State α) (rows cols : Nat) (positive : 0<rows*cols)
    (inputs : ∀ i, i<rows*cols → s.mem "input" i = .mk .int (signedValue (arangeWord i)))
    (outputs : ∀ i, i<rows*cols → s.mem "output" i = .mk .int (0 : Int)) (i : Nat) (hi : i<rows*cols) :
    (finalState (rows*cols) cols s).mem "output" i =
      .mk .int (signedValue (wordTarget rows cols i)) := by
  have columns : 0<cols := Nat.pos_of_mul_pos_left positive
  rw [final_output]
  have source := logical_readback (Cell.mk .int (0 : Int)) (rows*cols) cols columns
    (inputCells s) (s.mem "output") i hi
  have bound := target_congr_inside (Cell.mk .int (0 : Int)) (rows*cols) cols columns
    (inputCells s) (fun j ↦ Cell.mk .int (signedValue (arangeWord j)))
    (s.mem "output") (fun _ ↦ Cell.mk .int (0 : Int))
    (by intro j hj; simp [inputCells, inputs j hj])
    (by intro j hj; exact outputs j hj) i hi
  have mapped := congrFun (map_target (fun w : Word ↦ (Cell.mk .int (signedValue w) : Cell α))
    0 (rows*cols) cols arangeWord (fun _ ↦ 0)) i
  exact source.trans (bound.trans (by simpa [wordTarget, signedValue] using mapped.symm))


#axiomsClean initialized_cell
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.KernelResult
