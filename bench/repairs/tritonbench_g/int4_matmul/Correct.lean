import bench.repairs.tritonbench_g.int4_matmul.Proofs.LogicalCoordinates

namespace VeriTile.Bench.Repairs.Int4Matmul.Correct
open VeriTile.Triton KernelExecution
noncomputable section
open scoped BigOperators VeriTile.Triton.LogicalDomainIO

/-- Independent dequantized matrix product over all logical K entries.
Q and Z are the packed INT4 nibbles; scale is selected by k/group. The output
assembly only attaches the two logical matrix indices to the single channel. -/
def matrixProduct (s : Shape) (h : s.Valid) (xs : IO.MatrixInputs ℝ s)
    (row : Fin s.m) (col : Fin s.n) : ℝ :=
  ∑ k : Fin s.k, xs.a row k *
    ((IO.weightNibble s h xs k col-IO.zeroNibble s h xs k col : Int):ℝ) *
      xs.scale (IO.groupIndex s h k) col

/-- The complete FP32/INT32 source implements the independent dequantized
matrix product. Finite typed operands and the zeroed in-place output are
explicit in IO; every CTA, masked access, output and exterior cell is covered. -/
specification matmul_grid_real (x : Raw.Arguments) (geometry : Raw.Geometry x)
    (tile : Tile) (member : tile∈Host.sourceTiles) (rest : FP.Structural.Algebra ℝ)
    (lanes : List Lane) (complete : lanes.Perm (rawLanes x.shape tile)) :
    Spec.Real (IO.logicalIO x tile rest lanes ⊨{IO.matrixDomain x.shape}
      matrixProduct x.shape geometry.shape) :=
  IOProof.logical_real x geometry tile member rest lanes complete

#stmtSurfaceSubset matmul_grid_real ⊆
  [Spec.Real,IO.logicalIO,IO.matrixDomain,matrixProduct,LogicalDomainIO.ImplementsOn,
   IO.programIO,FP.Structural.ChannelValues,FP.Structural.PerAddressProgramIO.ImplementsOnCells,SourceGraph.Node,IO.Program,
   SourceGraph.instInhabitedNode,IO.MatrixInputs,IO.inputs,IO.outputs,
   Raw.Arguments,Raw.Arguments.shape,Raw.Geometry,Raw.Geometry.shape,Tile,Host.sourceTiles,
   FP.Structural.Algebra,Lane,rawLanes,List.Perm,Shape.Valid,Shape.m,Shape.n,realOps]
#specNonCircular Formula.mathematical avoiding
  [KernelSource.kernel,FP.Structural.exec,IO.programIO,IOProof.operands,
   KernelExecution.accumulation,KernelExecution.Raw.launch]
#axiomsClean matmul_grid_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int4Matmul.Correct
