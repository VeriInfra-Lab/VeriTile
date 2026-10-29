import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.NativeIO
import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Proofs.NativeImplementation

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph (Word ScalarOps)
noncomputable section

/-- The complete FP32 source observes the independent ordered attention/scatter
formula for every owner. Arbitrary jobs, duplicate writes, arbitrary input Exprs
and Q/K/V-output aliases are retained. Only the integer metadata is separated.
The existing scalar instruction laws are a primitive boundary, not an added
optimization rewrite, FP=Real assertion or experimental numerical assumption. -/
theorem kernel_native (c : Host.Call) (jobs : List (GridIndex c.grid))
    (separate : c.regions.out≠c.regions.start ∧ c.regions.out≠c.regions.sequence)
    (actual reference : ScalarOps) (laws : NativeInstructions.ScalarLaws actual reference) :
    (NativeInterface.nativeIO c jobs).ObservesOn (NativeInterface.observer c actual)
      (NativeArrays.metadata c) (NativeInterface.nativeTarget c jobs reference) :=
  NativeInterface.observes c jobs separate actual reference laws

#axiomsClean kernel_native
#stmtSurfaceSubset kernel_native ⊆ [ArrayProgramIO.ObservesOn,
  NativeInterface.nativeIO,NativeInterface.observer,NativeArrays.metadata,NativeInterface.nativeTarget,
  Host.Call,Host.Call.grid,Host.Call.regions,Program.Regions.out,Program.Regions.start,Program.Regions.sequence,
  GridIndex,ScalarOps,Word,NativeInstructions.ScalarLaws,NativePhysical.Expr,
  NativePhysical.instInhabitedExpr,RegionName]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel
