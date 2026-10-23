import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.KernelInterface
import bench.repairs.tritonbench_g.pow_scalar_tensor.Kernels
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Kernel
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeFP32
import VeriTile.Triton.Float.ProfiledMemory

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.RawNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer HostPlan SignedView
open VeriTile.Bench.TritonBenchG.PowScalarTensor Whole
noncomputable section
local instance : Inhabited Float32Bits := ⟨⟨0⟩⟩
/-- Complete native grid for either source variant over raw buffers and words.
The source pow primitive and its exact scalar ABI slot remain unchanged. -/
theorem raw_grid_fp32 (rest : Algebra Float32Bits) (base : ℝ) (baseWord : Float32Bits) (candidate : Bool)
    (input output : SignedView.View) {n inputCapacity outputCapacity : Nat} (hn : 0<n)
    (hin : input.Valid n inputCapacity) (hout : output.Valid n outputCapacity)
    (hstride : output.stride≠0) (hsep : SignedGrid.FootprintsDisjoint input output n)
    (initial : State Float32Bits) :
    ∃ final,runGrid (fp32Backend rest baseWord) base candidate (KernelInterface.arguments input output n) initial=some final ∧
      (∀ i,i<n → final.mem output.region (output.address i)=
        Cell.mk .fp32 (rest.binary (some .fp32) .real .pow baseWord
          ((initial.mem input.region (input.address i)).read .fp32))) ∧
      (∀ i,i<n → final.mem input.region (input.address i)=
        initial.mem input.region (input.address i)) ∧
      (∀ r a,(r≠output.region ∨ ∀ i,i<n → a≠output.address i) →
        final.mem r a=initial.mem r a) ∧
      (∀ i,i<n → input.address i < inputCapacity ∧ output.address i < outputCapacity) := by
  obtain ⟨final,ran,values,inputs,frame,bounds⟩ := grid_result (fp32Backend rest baseWord) base candidate
    (KernelInterface.buffer input) (KernelInterface.buffer output) hn input.stride output.stride
    hin hout hstride hsep initial
  refine ⟨final,ran,?_,inputs,frame,bounds⟩
  intro i hi
  simpa only [KernelInterface.view, KernelInterface.buffer, fp32_scalar] using values i hi

#axiomsClean raw_grid_fp32
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.RawNative
