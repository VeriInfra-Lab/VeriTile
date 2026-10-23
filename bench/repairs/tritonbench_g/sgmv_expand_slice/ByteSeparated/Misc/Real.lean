import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.Misc.Common
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.RealPrepared

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.Real
open VeriTile.Triton FP.Structural Contract
noncomputable section

/-- Direct source execution from original logical buffer observations. No
snapshot, allocation, successful-launch, or final-result premise is used. -/
theorem complete (t : Inputs) (s : Scalars) (lists : Lists t) (ok : Admitted t s spans lists)
    (rest : Algebra ℝ) (X : Nat→Nat→ℝ) (W : Nat→Nat→Nat→ℝ) (entry : Nat→Nat→ℝ) :
    ∃ final,
      runSourceJobs (SourceBody.install (RealBackend.algebra rest) RealBackend.dot)
        (abi t s) (args t s) (jobs (shape t s).columns (segments t s lists))
        (PreparedMemory.state (abi t s) (shape t s) X W entry (raw t s lists)
          (lists.lengths.drop s.batches.toNat))=some final ∧
      SourceSuffix.OutputMemory (abi t s) (args t s) (shape t s) final
        (mathematicalResult (shape t s) X W s.add (segments t s lists) entry) ∧
      SourceGrid.ReadFrame (args t s)
        (PreparedMemory.state (abi t s) (shape t s) X W entry (raw t s lists)
          (lists.lengths.drop s.batches.toNat)) final :=
  RealBackend.prepared_result rest (abi t s) (shape t s) (strides (shape t s)) X W entry
    (raw t s lists) (lists.lengths.drop s.batches.toNat) (segments t s lists)
    (validated ok) (xlayout _) (wlayout _) (ylayout _) (capacity ok)

#axiomsClean complete
#specNonCircular mathematicalResult avoiding [runSourceJobs,runRealJobs,FP.Structural.run]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.Real
