import bench.tritonbench_g.matmul_triton2.Proofs.SourceDomain
import bench.tritonbench_g.matmul_triton1.Proofs.FiniteSource

/-! The raw-body cast boundary follows from the existing contiguous invocation
and finite launch domain. All source configurations remain parameters. -/
namespace VeriTile.Bench.Repairs.MatmulTriton2.SourceRuntime
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

theorem SourceBounds.of_contiguous_ready (p : Core.Parameters)
    (canonical : VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteSource.Canonical p)
    (ready : VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex.Ready p) :
    SourceBounds p := by
  refine ⟨?_,?_,?_,?_,?_,?_⟩
  · intro name n member
    simp only [KernelOnly.scalarParameters,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with h | h | h | h | h | h | h | h | h
    all_goals cases h
    all_goals try simp only [canonical.inputRow,canonical.inputInner,canonical.weightInner,
      canonical.weightColumn,canonical.outputRow,canonical.outputColumn]
    all_goals have := ready.rows
    all_goals have := ready.columns
    all_goals have := ready.inner
    all_goals omega
  · have := ready.blockRows.2; omega
  · have := ready.blockColumns.2; omega
  · have := ready.blockInner.2; omega
  · have := ready.group.2; omega
  · change VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex.count p≤2^64
    have := ready.launch
    omega

#axiomsClean SourceBounds.of_contiguous_ready
end
end VeriTile.Bench.Repairs.MatmulTriton2.SourceRuntime
