import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Physical
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Kernel

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.KernelAccess
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer HostPlan SignedView
open VeriTile.Bench.TritonBenchG.PowScalarTensor Whole
noncomputable section

/-- Checked metadata supplies the actual signed32 indices/masks and every
active four-byte pointer access, by reusing the established source model. -/
theorem kernel_native_access (x y : ViewMetadata) (h : InputContract Bool.true x y) :
    NativeAccess (Physical.buffer x) (Physical.buffer y) (count x) (stride x) (stride y)
      (Physical.native_buffers x y) ∧
    (∀ i, i < count x → Physical.InsideBacking x i ∧ Physical.InsideBacking y i) := by
  have vy : ((Physical.buffer y).view (stride y)).Valid (count x) Physical.capacity := by
    rw [h.shape]; exact Physical.view_valid y h.output
  refine ⟨native_access (Physical.buffer x) (Physical.buffer y) h.positive h.extent (stride x) (stride y)
    (Physical.view_valid x h.input) vy (Physical.native_buffers x y),?_⟩
  intro i hi
  exact ⟨Physical.backing_bounds x h.input i hi,
    Physical.backing_bounds y h.output i (by simpa [←h.shape] using hi)⟩

#axiomsClean kernel_native_access
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.KernelAccess
