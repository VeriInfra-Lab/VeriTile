import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.Correct
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open VeriTile.Bench.TritonBenchG.PowScalarTensor Whole
noncomputable section

/-- Ordinary Real power domain, independent of the Python all-word FP32 contract. -/
def mathematical (base : ℝ) {n : Nat} (xs : Fin n → ℝ) : Prop :=
  ∀ i, powerDefined base (xs i)

abbrev powerIO := LogicalView.powerIO

/-- Complete actual source safety, successful grid execution, mathematical
power on every logical element, and the entire exterior typed-memory frame. -/
specification kernel_raw_correctness (rest : Algebra ℝ) (base : ℝ) (candidate : Bool)
    (input output : SignedView.View) {n inputCapacity outputCapacity : Nat} (hn : 0<n)
    (hin : input.Valid n inputCapacity) (hout : output.Valid n outputCapacity)
    (hstride : output.stride≠0) (hsep : SignedGrid.FootprintsDisjoint input output n) :
    Spec.Real (powerIO rest base candidate input output n ⊨{mathematical base}
      (fun xs i => Real.rpow base (xs i))) := by
  exact (LogicalReader.judgment_iff rest base candidate input output n).mpr
    (ReaderBefore.kernel_raw_correctness rest base candidate input output hn hin hout hstride hsep)

#specNonCircular mathematical avoiding [sourceKernel,runGrid,FP.Structural.exec]
#axiomsClean kernel_raw_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.Correct
