import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.NativeObservation
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.KernelCompatibility
import VeriTile.Meta.Specification

/-! Native output observations for the entire supplied launch order, including
repetitions. NativeIO.lean and NativeArrays.lean expose the unchanged FP32
source, typed named arrays, finite gather windows and alias-aware recurrence.
Only the established primitive ScalarLaws are assumed; no floating rewrite or
native-to-Real equality is asserted, so there is no transformation assumption
for #print_fp_assumptions to report. -/
namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section

theorem context_attn_llama_native (c : Host.Call) (order : List (GridIndex c.grid))
    (separate : c.regions.start≠c.regions.out ∧ c.regions.sequence≠c.regions.out ∧ c.regions.prompt≠c.regions.out)
    (actual referenceOps : NativeInstructions.ScalarOps)
    (localLaws : NativeInstructions.ScalarLaws actual referenceOps) :
    (NativeInterface.nativeIO c order).ObservesOn (NativeInterface.observer c order actual)
      (NativeArrays.metadata c) (NativeInterface.nativeFormula c order referenceOps) :=
  NativeInterface.native_observes c order separate actual referenceOps localLaws

#stmtSurfaceSubset context_attn_llama_native ⊆
  [RegionName,NativeGraph.instInhabitedExpr,Host.Call,Host.Call.grid,Host.Call.regions,GridIndex,Program.Regions.start,Program.Regions.out,
   Program.Regions.sequence,Program.Regions.prompt,NativeInstructions.ScalarOps,NativeInstructions.ScalarLaws,
   NativeInstructions.Expr,NativeInstructions.Word,NativeInterface.nativeIO,
   ArrayProgramIO.ObservesOn,NativeInterface.observer,NativeArrays.metadata,NativeInterface.nativeFormula]
#axiomsClean context_attn_llama_native
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
