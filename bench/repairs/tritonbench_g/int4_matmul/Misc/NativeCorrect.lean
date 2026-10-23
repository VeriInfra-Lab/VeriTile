import bench.repairs.tritonbench_g.int4_matmul.Proofs.IOImplements
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Native

namespace VeriTile.Bench.Repairs.Int4Matmul.FPEquiv
open VeriTile.Triton KernelExecution
noncomputable section

/-- Native observation of the same complete source, with its original scalar
and fixed-K8 ISA model, live accumulator, K padding and per-address atomic
order. This is the retained source-to-native result, not a new FP rewrite.
All floating words are admitted; scalar/ISA models are primitive boundaries,
not experimental numerical assumptions. -/
theorem matmul_grid_native (x : Raw.Arguments) (geometry : Raw.Geometry x)
    (tile : Tile) (member : tile∈Host.sourceTiles)
    (instructions : Native.Instructions) (isa : Native.ISA) (rest : FP.Structural.Algebra Word)
    (lanes : List Lane) (complete : lanes.Perm (rawLanes x.shape tile)) :
    (IO.programIO x tile (Native.ops instructions isa) rest lanes).ImplementsOnCells
      (IO.domain x.shape (Native.ops instructions isa))
      (Formula.nativeOutput x.shape tile (Native.ops instructions isa) lanes) :=
  IOProof.implements x geometry tile member (Native.ops instructions isa) rest lanes complete

#stmtSurfaceSubset matmul_grid_native ⊆
  [IO.programIO,IO.domain,Formula.nativeOutput,
   FP.Structural.PerAddressProgramIO.ImplementsOnCells,SourceGraph.Node,IO.Program,
   SourceGraph.instInhabitedNode,
   Raw.Arguments,Raw.Arguments.shape,Raw.Geometry,Tile,Host.sourceTiles,
   FP.Structural.Algebra,Word,Native.Instructions,Native.ISA,Native.ops,Lane,rawLanes,List.Perm]
#specNonCircular Formula.nativeOutput avoiding
  [KernelSource.kernel,FP.Structural.exec,IO.programIO,IOProof.operands,
   KernelExecution.accumulation,KernelExecution.Raw.launch,partialValue]
#specNonCircular IO.domain avoiding
  [KernelSource.kernel,FP.Structural.exec,FP.Structural.TraceSafe,
   FP.Structural.PerAddressProgramIO.ImplementsOnCells,SourceGraph.Node,IO.Program,
   SourceGraph.instInhabitedNode,Formula.nativeOutput]
#axiomsClean matmul_grid_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int4Matmul.FPEquiv
