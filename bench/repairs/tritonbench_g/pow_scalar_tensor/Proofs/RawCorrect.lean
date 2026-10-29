import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.KernelInterface
import bench.repairs.tritonbench_g.pow_scalar_tensor.Kernels
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Kernel
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeReal

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.RawCorrect
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer HostPlan SignedView
open VeriTile.Bench.TritonBenchG.PowScalarTensor Whole
noncomputable section

/-- Complete raw grid: ordinary Real power domain, all output cells, input
preservation, exterior frame and active address bounds. -/
theorem kernel_raw_correctness (rest : Algebra ℝ) (base : ℝ) (candidate : Bool)
    (input output : SignedView.View) {n inputCapacity outputCapacity : Nat} (hn : 0<n)
    (hin : input.Valid n inputCapacity) (hout : output.Valid n outputCapacity)
    (hstride : output.stride≠0) (hsep : SignedGrid.FootprintsDisjoint input output n)
    (initial : State ℝ)
    (_mathematical : ∀ i,i<n → powerDefined base
      ((initial.mem input.region (input.address i)).read .fp32)) :
    ∃ final,runGrid (realBackend rest) base candidate (KernelInterface.arguments input output n) initial=some final ∧
      (∀ i,i<n → final.mem output.region (output.address i)=
        Cell.mk .fp32 (Real.rpow base ((initial.mem input.region (input.address i)).read .fp32))) ∧
      (∀ i,i<n → final.mem input.region (input.address i)=
        initial.mem input.region (input.address i)) ∧
      (∀ r a,(r≠output.region ∨ ∀ i,i<n → a≠output.address i) →
        final.mem r a=initial.mem r a) ∧
      (∀ i,i<n → input.address i < inputCapacity ∧ output.address i < outputCapacity) := by
  obtain ⟨final,ran,values,inputs,frame,bounds⟩ := grid_result (realBackend rest) base candidate
    (KernelInterface.buffer input) (KernelInterface.buffer output) hn input.stride output.stride
    hin hout hstride hsep initial
  refine ⟨final,ran,?_,inputs,frame,bounds⟩
  intro i hi
  simpa only [KernelInterface.view, KernelInterface.buffer, real_scalar] using values i hi

#axiomsClean kernel_raw_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.RawCorrect
