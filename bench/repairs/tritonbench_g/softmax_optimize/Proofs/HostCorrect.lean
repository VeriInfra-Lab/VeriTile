import bench.repairs.tritonbench_g.softmax_optimize.Kernels
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.GridReal
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Pointers
namespace VeriTile.Bench.Repairs.SoftmaxOptimize.HostCorrect
open VeriTile Triton
open scoped BigOperators
noncomputable section

/-- Independent full-row mathematical softmax. -/
def target (input : Fin M → Fin N → ℝ) (row : Fin M) (column : Fin N) : ℝ :=
  Real.exp (input row column) / ∑ j : Fin N, Real.exp (input row j)

specification softmax_real (x y : TensorMetadata) (admitted : InputContract x)
    (allocated : Allocated x y) (initial : BlockState)
    (input : Fin (rows x) → Fin (columns x) → ℝ)
    (loads : ∀ row column, initial.readMem "x" (row.val*columns x+column.val)=input row column) :
    Spec.Real (check x=.ok () ∧ y.shape=x.shape ∧ y.device=x.device ∧
      ∃ final, GridReal.launch "out" "x" (rows x) (columns x) initial=some final ∧
        (∀ row column, final.readMem "out" (row.val*columns x+column.val)=target input row column) ∧
        (∀ (region : RegionName) offset, region≠"out" ∨ x.numel≤offset → final.mem region offset=initial.mem region offset) ∧
        final.undef=initial.undef) := by
  obtain ⟨final,he,hv,hf,hu⟩ := GridReal.launch_io "out" "x" (rows x) (columns x) (by decide) initial
  refine ⟨(check_iff x).mpr admitted,allocated.1,allocated.2.1,final,he,?_,?_,hu⟩
  · intro row column
    rw [hv row column]
    unfold GridReal.rowSpec target
    simp_rw [loads row]
  · intro region offset outside
    apply hf region offset
    simpa only [count_two x admitted.2.2.2.1] using outside

#specNonCircular target avoiding [GridReal.launch,GridReal.runRows,exec]
#axiomsClean softmax_real
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.HostCorrect
