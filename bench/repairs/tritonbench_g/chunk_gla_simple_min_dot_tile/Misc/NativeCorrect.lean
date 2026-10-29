import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.IONative
import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.KernelNativeResults

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.FPEquiv
open VeriTile Triton FP.Structural
open Host
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions (Graph)
noncomputable section

/-- The same complete source schedule, with arbitrary symbolic inputs and
the original FP32 load, accumulator, gate, cast and runtime-scale semantics. -/
def nativeIO (c : Shape) : ProgramIO NativeGraph.Expr NativeGraph.Expr := PublicIO.nativeIO c

/-- Preserve the existing lower-to-Graph observation and its runtime FP32
scale word; every dot accumulator becomes an ordered scalar-FMA chain. -/
def observation (c : Shape) (scale : BitVec 32) :
    (o : Fin (nativeIO c).outputs.length) → TileIndex ((nativeIO c).outputs[o]).shape →
      NativeGraph.Expr → Graph := fun _ _ => ScalarGraph.lower scale

/-- Complete source-to-native observation. The five fixed primitive families
remain a model boundary in ScalarGraph.WordPrimitives; no new numerical law
or candidate kernel is introduced by this interface. -/
theorem whole_kernel (c : Shape) (bt : 0<c.BT) (scale : BitVec 32) :
    (nativeIO c).Observes (observation c scale) (PublicNative.formula c bt scale) :=
  PublicIO.native_observes c bt scale

#axiomsClean whole_kernel
#stmtSurfaceSubset whole_kernel ⊆
  [ProgramIO.Observes,nativeIO,observation,PublicNative.formula,Shape,Shape.BT,
   NativeGraph.Expr,NativeGraph.instInhabitedExpr,Graph]
#specNonCircular PublicNative.formula avoiding
  [kernel,FPGrid.run,FPGrid.reference,FPValue.outputValue,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.FPEquiv
