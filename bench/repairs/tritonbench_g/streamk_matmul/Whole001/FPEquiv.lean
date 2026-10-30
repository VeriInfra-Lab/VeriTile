import bench.repairs.tritonbench_g.streamk_matmul.Whole001.ABI
namespace VeriTile.Bench.Repairs.StreamK.Whole001
open Source
noncomputable section

/-- Whole returned FP32 payload graph. Numerical premises are fixed-k8 MMA,
scalar half conversion and scalar atomic add only. -/
theorem whole_returned_native (p : Parameters) (actual : Native.Instructions)
    (ref : Native.Reference) (numeric : Native.Numerical actual ref)
    (initial : State Native.Word p exterior)
    (first : Schedule.Order (firstWave p (Native.operations actual)).length)
    (reduce : Schedule.CTAInterleaving p)
    (last : Schedule.Order (fullWave p (Native.operations actual)).length) :
    let final:=Schedule.call p (Native.operations actual) first reduce last initial
    (∀ cell : Returned.Cell p, final.output (Returned.index p cell)=
      Returned.value p (Native.operations (Native.referenceInstructions ref)) initial.inputA initial.inputB cell) ∧
    final.inputA=initial.inputA ∧ final.inputB=initial.inputB ∧ final.exterior=initial.exterior := by
  dsimp only
  rw [Schedule.call_eq]
  refine ⟨?_,call_frame p (Native.operations actual) initial⟩
  intro cell
  exact Native.returned_fp p actual ref numeric initial cell

#axiomsClean whole_returned_native
end
end VeriTile.Bench.Repairs.StreamK.Whole001
