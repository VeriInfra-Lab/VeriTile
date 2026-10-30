import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.RealWhole
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.Correct
/-- Independent full returned attention formula, source branch, and finite frame. -/
abbrev producer_correct := @RealWhole.producer
/-- Full standalone consumer contraction over caller-supplied attention/history. -/
abbrev consumer_correct := @RealWhole.consumer
/-- Complete source mathematics over raw regional input values and geometry. -/
theorem kernel_raw_correct (mode : Admission.Mode) (c : Host.Dimensions)
    (scale : ℝ) (initial : VeriTile.Triton.RegionName → Nat → ℝ) :
    ∃ final,KernelReal.sourceExecution mode c scale (KernelReal.entry initial) final ∧
      KernelReal.RawPost mode c scale (KernelReal.entry initial) final :=
  KernelReal.raw_complete mode c scale initial

#axiomsClean kernel_raw_correct
#axiomsClean producer_correct
#axiomsClean consumer_correct
end VeriTile.Bench.Repairs.ChunkGlaFwd.Correct
