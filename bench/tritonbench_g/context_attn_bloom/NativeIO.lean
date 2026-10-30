import bench.tritonbench_g.context_attn_bloom.IO
import bench.tritonbench_g.context_attn_bloom.Proofs.NativePlan

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.NativeIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Exact original FP32 source and complete metadata-selected serial grid.
Symbolic payloads keep all source reads, including raw-tag default values. -/
def io (g : Raw.Geometry) (r : VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program.Regions)
    (extent : RegionBounds) : ProgramIO Native.Expr Native.Expr :=
  KernelIO.io g r extent Native.scalar Native.dots

/-- The existing SM80 plan depends only on source topology and geometry.
The stored lane owner is the logical output column, exactly as before. -/
def observe (g : Raw.Geometry) (r : VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program.Regions)
    (extent : RegionBounds) (ops : Native.WordISA) :
    (o : Fin (io g r extent).outputs.length)→TileIndex ((io g r extent).outputs[o]).shape→Native.Expr→BitVec 32
  | ⟨0,_⟩,i,e=>(Native.lower (KernelNative.plan g.arguments) e (KernelIO.coordinate g i.1).2.2.1.val).eval ops.interpret

/-- Native observation of the independent ordered tile formula. Primitive
word instructions are explicit; no FP-to-Real or algebraic rewrite is assumed. -/
def formula (g : Raw.Geometry) (r : VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program.Regions)
    (extent : RegionBounds) (ops : Native.WordISA) (xs : (io g r extent).Inputs) :
    (io g r extent).ObservedOutputs (BitVec 32) :=
  (io g r extent).mapOutputs (observe g r extent ops)
    (KernelIO.formula g r extent Native.scalar Native.dots xs)

end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.NativeIO
