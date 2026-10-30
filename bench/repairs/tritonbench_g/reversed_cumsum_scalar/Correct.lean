import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.LogicalReader
import VeriTile.Meta.Specification

/-! Independent suffix sum for each admitted source row. IO.lean exposes the
original strided windows and complete source loop; physical placement and
raw execution corollaries remain in Proofs/KernelCompatibility. -/
namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.Correctness
open VeriTile.Triton Interface
open scoped VeriTile.Triton.LogicalIO
noncomputable section

/-- Every output of the selected row is its finite suffix sum. The shared
judgment includes actual masked-access safety and the exact exterior frame. -/
specification reversed_cumsum_scalar_correctness (h : Call) (layout : Geometry.OutputLayout h)
    (c : Config) (chosen : c∈configurations) (row : Fin (h.B*h.H)) :
    Spec.Real (logicalIO h c row ⊨ fun xs t =>
      (∑ k : Fin h.T, if t.val ≤ k.val then xs k else 0 : ℝ)) :=
  (IOComplete.logical_iff h c row).mpr (IOComplete.implements h layout c chosen row)

#specNonCircular reference avoiding [exec,RealStorage.execute,Kernels.source,Kernels.indexCommuted]
#stmtSurfaceSubset reversed_cumsum_scalar_correctness ⊆
  [Spec.Real,Call,Call.B,Call.H,Call.T,Geometry.OutputLayout,Config,configurations,
    logicalIO,LogicalIO.Implements,Inputs,io,AlgorithmProgramIO.Implements]
#axiomsClean reversed_cumsum_scalar_correctness
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.Correctness
