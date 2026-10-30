import bench.tritonbench_g.matmul_triton1.Proofs.LogicalIO

/-! This entry proves the selected compiler graph. The separate
`pythonBody` / `pythonSource` constructors in Kernels.lean retain the original
standalone dot followed by addition; no source-to-compiler FP equivalence is
asserted by this result. -/

namespace VeriTile.Bench.Repairs.MatmulTriton1
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section


/-- Complete selected FP32 source grid and the independent full-K product.
All input, storage, geometry, separation, safety and frame conditions remain. -/
specification matmul_triton1_real (rest : Algebra ℝ) (p : Core.Parameters)
    (canonical : FiniteSource.Canonical p) (ready : FiniteIndex.Ready p)
    (storage : FiniteIndex.StorageBounds p) (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    Spec.Real (MatmulIO.logicalIO rest Core.realPrimitives p ⊨ fun ⟨a,b⟩ row col =>
      ∑ k : Fin p.innerSize, a row k * b k col) :=
  LogicalIOProof.implements rest p canonical ready storage geometry ha hb

#stmtSurfaceSubset matmul_triton1_real ⊆
  [matmul_triton1_real.match_1,Spec.Real,LogicalProgramIO.Implements,MatmulIO.logicalIO,Core.realPrimitives,Core.Parameters,Core.Parameters.rows,Core.Parameters.columns,Core.Parameters.innerSize,Core.Parameters.input,Core.Parameters.weight,Core.Parameters.output,FP.Structural.Algebra,FiniteSource.Canonical,FiniteIndex.Ready,FiniteIndex.StorageBounds,Geometry.Valid,Core.geometry,RegionName]
#axiomsClean matmul_triton1_real
#specNonCircular MatmulIO.realTarget avoiding [Core.source,Kernel.source,Core.grid,FiniteSource.grid,FP.Structural.exec,Core.logical]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulTriton1
