import bench.tritonbench_g.pow_scalar_tensor.Proofs.IOResults

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.ReaderBefore
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.ProgramIO
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open VeriTile.Bench.TritonBenchG.PowScalarTensor Whole
noncomputable section

/-- Ordinary Real power domain, independent of the Python all-word FP32 contract. -/
def mathematical (base : ℝ) {input : SignedView.View} {n : Nat}
    (xs : Interface.Inputs ℝ input n) : Prop :=
  ∀ i : Fin n,powerDefined base (Interface.inputValue xs i)

/-- Full signed-stride input/output spans, exact capped CTA schedule and either
actual source variant; all load/cast/power/store precision remains in Kernels. -/
def powerIO (rest : Algebra ℝ) (base : ℝ) (candidate : Bool)
    (input output : SignedView.View) (n : Nat) : ProgramIO ℝ ℝ :=
  Interface.programIO (realBackend rest) base candidate input output n

/-- Complete actual source safety, successful grid execution, mathematical
power on every logical element, and the entire exterior typed-memory frame. -/
theorem kernel_raw_correctness (rest : Algebra ℝ) (base : ℝ) (candidate : Bool)
    (input output : SignedView.View) {n inputCapacity outputCapacity : Nat} (hn : 0<n)
    (hin : input.Valid n inputCapacity) (hout : output.Valid n outputCapacity)
    (hstride : output.stride≠0) (hsep : SignedGrid.FootprintsDisjoint input output n) :
    Spec.Real (powerIO rest base candidate input output n ⊨{mathematical base}
      (fun xs => Interface.result (fun i => Real.rpow base (Interface.inputValue xs i)))) := by
  exact Interface.real rest base candidate input output hn hin hout hstride hsep

end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.ReaderBefore
