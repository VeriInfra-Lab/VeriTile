import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.ReaderFormat
import VeriTile.Meta.Specification

/-! Independent suffix sums for every active time/feature output of one kernel
program. IO.lean exposes the source, precision interpretation, windows, tail
mask and launch axes; physical placement results are in TensorBridge. -/
namespace VeriTile.Bench.Repairs.ReversedCumsum.Correctness
open VeriTile.Triton Interface
open scoped BigOperators VeriTile.Triton.LogicalIO
noncomputable section

/-- The selected row and feature tile implement ordinary suffix sums. The
shared judgment also connects all actual accesses and the exact exterior frame. -/
specification kernel_correct (h : Call) (distinct : h.input≠h.output)
    (c : Config) (chosen : c∈configurations) (program : Geometry.Program h) :
    Spec.Real (suffixIO h c program ⊨ fun xs time feature =>
      ∑ t : Fin h.T, if time ≤ t then xs t feature else 0) :=
  ReaderFormat.correct h distinct c chosen program

#specNonCircular reference avoiding [Kernels.source,Kernels.compareRight,exec]
#stmtSurfaceSubset kernel_correct ⊆
  [Spec.Real,RegionName,Call,Call.input,Call.output,Call.T,Config,configurations,Geometry.Program,
    suffixIO,LogicalIO.Implements,Interface.Matrix,BS,Inputs,AlgorithmProgramIO.Implements,io]
#axiomsClean kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReversedCumsum.Correctness
