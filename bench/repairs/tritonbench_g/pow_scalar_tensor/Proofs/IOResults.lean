import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.IOInterface
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeReal
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeFP32

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.Interface
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open VeriTile.Bench.TritonBenchG.PowScalarTensor Whole
noncomputable section
local instance : Inhabited Float32Bits := ⟨⟨0⟩⟩

theorem real (rest : Algebra ℝ) (base : ℝ) (candidate : Bool)
    (input output : SignedView.View) {n inputCapacity outputCapacity : Nat} (hn : 0<n)
    (hin : input.Valid n inputCapacity) (hout : output.Valid n outputCapacity)
    (stride : output.stride≠0) (separate : SignedGrid.FootprintsDisjoint input output n) :
    (programIO (realBackend rest) base candidate input output n).ImplementsOn
      (fun xs => ∀ i : Fin n,powerDefined base (inputValue xs i))
      (fun xs => result (fun i => Real.rpow base (inputValue xs i))) := by
  have proved := implements (realBackend rest) base candidate input output hn hin hout stride separate
  intro xs _ bounds bounded initial binding
  simpa only [real_scalar] using proved xs trivial bounds bounded initial binding

theorem native (rest : Algebra Float32Bits) (base : ℝ) (baseWord : Float32Bits) (candidate : Bool)
    (input output : SignedView.View) {n inputCapacity outputCapacity : Nat} (hn : 0<n)
    (hin : input.Valid n inputCapacity) (hout : output.Valid n outputCapacity)
    (stride : output.stride≠0) (separate : SignedGrid.FootprintsDisjoint input output n) :
    (programIO (fp32Backend rest baseWord) base candidate input output n).Implements
      (fun xs => result (fun i => rest.binary (some .fp32) .real .pow baseWord (inputValue xs i))) := by
  simpa only [fp32_scalar] using
    implements (fp32Backend rest baseWord) base candidate input output hn hin hout stride separate
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.Interface
