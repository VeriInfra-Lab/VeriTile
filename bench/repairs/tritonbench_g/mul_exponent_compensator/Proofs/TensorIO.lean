import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.PhysicalIO
import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.TensorIO
open VeriTile.Triton FP.Structural KernelTensor
open VeriTile.Torch.FP32 (Tensor TensorLaws)
open VeriTile.Bench.Repairs.TritonMul2 (Memory)
open TritonMul2.Host (logical logical_contiguous)
noncomputable section

/-- Apply the source IO/finite-address transport at the original actual Tensor
buffers, retaining all logical outputs, input preservation and exterior words. -/
theorem execute (M : Algebra α) [Inhabited α] (f : α → α) (identify : ∀ x,scalar M x=f x)
    (x : Tensor) (lx : TensorLaws x) (current B : Nat)
    (admit : InputContract x.metadata current B) (p : Buffers x lx current B admit)
    (before : Memory α) (order : List Nat)
    (schedule : order.Perm (List.range (grid x.metadata.numel B))) :
    ∃ final,Native.sourceLaunch ⟨f⟩ (layout p) order before=some final ∧
      (∀ i,i<x.metadata.numel → logical final p.output i=f (logical before x i)) ∧
      Frame p before final ∧
      (∀ i,i<x.metadata.numel → logical final x i=logical before x i) := by
  obtain ⟨final,execution,values,frame⟩ := PhysicalIO.complete M f identify (layout p)
    admit.block admit.grid (allocation p) order schedule before
  refine ⟨final,execution,?_,frame,input_frame p before _ frame⟩
  intro i hi
  rw [logical_contiguous _ p.output p.laws p.contiguous i (by rw [p.count];exact hi),
    logical_contiguous before x lx admit.contiguous i hi]
  exact values i hi

#axiomsClean execute
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.TensorIO
