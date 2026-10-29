import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeImplements
import bench.repairs.tritonbench_g.chunk_gla_fwd.Misc.RawFP

/-! Native observation of the original FP32 source, with runtime scale word,
reduction layout and local seven-port instruction ISA explicit. NativeIO
exposes the full original schedules and finite typed channels. Raw FP32 words
are unrestricted. This is an independent native result, not a kernel rewrite
or a native-to-Real equality; no transformation assumption is introduced. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeCorrect
open VeriTile Triton FP.Structural NativeInstructions
open NativePorts (Word)
noncomputable section

/-- Complete producer, preserving the source casts, TF32 options, ordered
reductions and both K branches. Geometry-only masks partition the actual
Structural storage tags of one returned FP32 attention tensor. -/
theorem producer_native_observation (c : Host.Dimensions)
    (I : ISA Word) (w : KernelProfile.Profile) (scale : Word) :
    (NativeIO.producer c I w scale).Implements (NativeIO.attention c I w scale) :=
  NativeImplements.producer_correct c I w scale

/-- Complete standalone consumer on independently supplied attention/history,
with every returned FP32 word and exact exterior cell frame. -/
theorem consumer_native_observation (c : Host.Dimensions)
    (I : ISA Word) (w : KernelProfile.Profile) (scale : Word) :
    (NativeIO.consumer c I w scale).Implements (NativeIO.output c I w scale) :=
  NativeImplements.consumer_correct c I w scale

#axiomsClean producer_native_observation
#axiomsClean consumer_native_observation
#stmtSurfaceSubset producer_native_observation ⊆ [Host.Dimensions,ISA,Word,KernelProfile.Profile,
  FillProgramIO.Implements,NativeIO.producer,NativeIO.attention,NativePorts.Expr,NativeStaging.instInhabitedExpr]
#stmtSurfaceSubset consumer_native_observation ⊆ [Host.Dimensions,ISA,Word,KernelProfile.Profile,
  FillProgramIO.Implements,NativeIO.consumer,NativeIO.output,NativePorts.Expr,NativeStaging.instInhabitedExpr]
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeCorrect
