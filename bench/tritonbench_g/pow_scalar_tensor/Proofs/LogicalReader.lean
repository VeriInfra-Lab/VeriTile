import bench.tritonbench_g.pow_scalar_tensor.IO
import bench.tritonbench_g.pow_scalar_tensor.Proofs.ReaderBefore
namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.LogicalReader
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open VeriTile.Bench.TritonBenchG.PowScalarTensor Whole
noncomputable section
/-- Equality of complete judgments, including the Real power domain, source
safety, successful whole-grid execution and exact exterior frame. -/
theorem judgment_iff (rest : Algebra ℝ) (base : ℝ) (candidate : Bool)
    (input output : SignedView.View) (n : Nat) :
    (LogicalView.powerIO rest base candidate input output n).ImplementsOn
      (fun xs => ∀ i, powerDefined base (xs i)) (fun xs i => Real.rpow base (xs i)) ↔
    (ReaderBefore.powerIO rest base candidate input output n).ImplementsOn
      (ReaderBefore.mathematical base)
      (fun xs => Interface.result (fun i => Real.rpow base (Interface.inputValue xs i))) := Iff.rfl
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.LogicalReader
