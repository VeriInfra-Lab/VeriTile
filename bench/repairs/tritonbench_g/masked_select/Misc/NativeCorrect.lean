import bench.repairs.tritonbench_g.masked_select.IO
import bench.repairs.tritonbench_g.masked_select.Proofs.PublicIO

/-! Actual twenty autotune source configurations, including complete grids.
Payloads are copied opaquely: all native FP32 words, NaNs and signed zeros are
admitted. No floating arithmetic or numerical transformation law is involved.
-/
namespace VeriTile.Bench.Repairs.MaskedSelect.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MaskedSelect TypedCopy Whole
noncomputable section

/-- Both actual configuration grids are safe and succeed. Active output cells
agree, and both preserve the full exterior frame from the same prepared inputs. -/
theorem masked_select_configuration_io_equiv {α : Type} [Inhabited α]
    (rest : Algebra α) (p : Parameters) (left right : Config)
    (hl : left ∈ configs) (hr : right ∈ configs) (separate : KernelSeparate p) :
    (IO.selectionIO rest p left).EquivalentOnCells (IO.candidate rest p left right)
      (IO.Prefixes p) :=
  PublicIO.equivalence rest p left right hl hr separate

#stmtSurfaceSubset masked_select_configuration_io_equiv ⊆
  [FP.Structural.Algebra, Parameters, Config, configs, KernelSeparate,
   IO.selectionIO, IO.candidate, ProgramIO.EquivalentOnCells, IO.Prefixes]
-- The shared structural judgment has no floating transformation parameter.
-- Exact old full-memory equality remains in Proofs/ConfigurationEquivalence.
#axiomsClean masked_select_configuration_io_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedSelect.FPEquiv
