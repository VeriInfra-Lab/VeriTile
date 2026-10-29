import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.IOResults
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.KernelInterface
import bench.repairs.tritonbench_g.pow_scalar_tensor.Kernels
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Kernel
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeFP32
import VeriTile.Triton.Float.ProfiledMemory

/-! Auxiliary native grid observation for either source variant. The source index-commutation equivalence is in FPEquiv. -/
namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.FPEquiv
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer HostPlan SignedView
open VeriTile.Bench.TritonBenchG.PowScalarTensor Whole
open scoped VeriTile.Spec VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section
local instance kernelFloat32BitsInhabited : Inhabited Float32Bits := ⟨⟨0⟩⟩

/-- Complete native grid for either source variant. The unchanged opaque power
primitive receives the same runtime scalar ABI word and every logical input. -/
def nativeIO (rest : Algebra Float32Bits) (base : ℝ) (baseWord : Float32Bits) (candidate : Bool)
    (input output : SignedView.View) (n : Nat) : ProgramIO Float32Bits Float32Bits :=
  Interface.programIO (fp32Backend rest baseWord) base candidate input output n

theorem raw_grid_fp32 (rest : Algebra Float32Bits) (base : ℝ) (baseWord : Float32Bits) (candidate : Bool)
    (input output : SignedView.View) {n inputCapacity outputCapacity : Nat} (hn : 0<n)
    (hin : input.Valid n inputCapacity) (hout : output.Valid n outputCapacity)
    (hstride : output.stride≠0) (hsep : SignedGrid.FootprintsDisjoint input output n) :
    nativeIO rest base baseWord candidate input output n ⊨ fun xs =>
      Interface.result (fun i => rest.binary (some .fp32) .real .pow baseWord (Interface.inputValue xs i)) := by
  exact Interface.native rest base baseWord candidate input output hn hin hout hstride hsep

#stmtSurfaceSubset raw_grid_fp32 ⊆
  [FP.Structural.Algebra,Float32Bits,SignedView.View,SignedView.View.Valid,SignedView.View.stride,SignedView.View.region,
   SignedGrid.FootprintsDisjoint,nativeIO,ProgramIO.Implements,ProgramIO.Inputs,
   Interface.result,Interface.inputValue,Algebra.binary,ComputeDType,ComputeDType.fp32,FloatDType.real,
   FP.Structural.Binary.pow,kernelFloat32BitsInhabited]

#axiomsClean raw_grid_fp32

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.FPEquiv
