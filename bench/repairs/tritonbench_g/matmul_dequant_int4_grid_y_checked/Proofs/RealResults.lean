import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.KernelInterface
import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Spec

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.RealResults
open VeriTile.Triton FP.Structural
noncomputable section

def kernel_mathematical (f : Source.Profile) (p : Source.Parameters) (initial : State ℝ)
    (x : KernelGrid.Coordinate p) : ℝ :=
  (KernelInterface.difference p initial x : ℝ)*KernelInterface.scale f p initial x

theorem kernel_real (f : Source.Profile) (p : Source.Parameters)
    (valid : KernelGrid.Valid p) (initial : State ℝ) :
    ∃ final,KernelGrid.run f Spec.real p initial=some final ∧
      (∀ x : KernelGrid.Coordinate p,
        final.mem p.output (KernelGrid.address p x)=Source.cell f (kernel_mathematical f p initial x)) ∧
      (∀ r o,(r≠p.output ∨ ∀ x : KernelGrid.Coordinate p,o≠KernelGrid.address p x) →
        final.mem r o=initial.mem r o) := by
  obtain ⟨final,ran,values,frame⟩ := KernelGrid.whole f Spec.real p valid initial
  refine ⟨final,ran,?_,?_⟩
  · intro x
    rw [values x trivial]
    congr 1
  · intro r o outside
    exact frame r o (outside.imp_right (fun h x _ => h x))

#specNonCircular kernel_mathematical avoiding [KernelGrid.run,Source.source,Source.execution,KernelGrid.result]
#axiomsClean kernel_real
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.RealResults
