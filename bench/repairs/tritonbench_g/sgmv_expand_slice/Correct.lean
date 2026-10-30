import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Direct.Real

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.Correct
open VeriTile.Triton FP.Structural Contract
noncomputable section

/-- Executable admission checks imply complete typed source execution and the
independent ordinary K-term formula in original segment order. -/
theorem checked_real (t : Inputs) (s : Scalars) (lists : Lists t) (ok : Admitted t s lists)
    (rest : Algebra ℝ) (X : Nat→Nat→ℝ) (W : Nat→Nat→Nat→ℝ) (entry : Nat→Nat→ℝ) :
    ∃ final,
      runSourceJobs (SourceBody.install (RealBackend.algebra rest) RealBackend.dot)
        (abi t s) (Direct.args t s) (jobs (shape t s).columns (segments t s lists))
        (PreparedMemory.state (abi t s) (shape t s) X W entry (raw t s lists)
          (lists.lengths.drop s.batches.toNat))=some final ∧
      SourceSuffix.OutputMemory (abi t s) (Direct.args t s) (shape t s) final
        (mathematicalResult (shape t s) X W s.add (segments t s lists) entry) ∧
      SourceGrid.ReadFrame (Direct.args t s)
        (PreparedMemory.state (abi t s) (shape t s) X W entry (raw t s lists)
          (lists.lengths.drop s.batches.toNat)) final :=
  Direct.Real.complete t s lists ok rest X W entry

#axiomsClean checked_real
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.Correct
