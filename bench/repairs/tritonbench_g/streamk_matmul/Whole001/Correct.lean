import bench.repairs.tritonbench_g.streamk_matmul.Whole001.ABI
namespace VeriTile.Bench.Repairs.StreamK.Whole001
open Source
noncomputable section

/-- Whole repaired FP32 task under ordinary finite-view/compiler/runtime ABI:
actual ordered launch state, arbitrary source CTA interleavings, all returned
M*N cells, input and exterior frame, and independent Real matmul semantics. -/
theorem whole_returned_real (p : Parameters) (initial : State ℝ p exterior)
    (first : Schedule.Order (firstWave p Arithmetic.real).length)
    (reduce : Schedule.CTAInterleaving p)
    (last : Schedule.Order (fullWave p Arithmetic.real).length) :
    let final:=Schedule.call p Arithmetic.real first reduce last initial
    (∀ cell : Returned.Cell p, final.output (Returned.index p cell)=
      Reference.matrixProduct p initial.inputA initial.inputB cell.1 cell.2) ∧
    final.inputA=initial.inputA ∧ final.inputB=initial.inputB ∧ final.exterior=initial.exterior := by
  dsimp only
  rw [Schedule.call_eq]
  refine ⟨?_,call_frame p Arithmetic.real initial⟩
  intro cell
  exact Returned.real_contract p initial cell

#axiomsClean whole_returned_real
end
end VeriTile.Bench.Repairs.StreamK.Whole001
