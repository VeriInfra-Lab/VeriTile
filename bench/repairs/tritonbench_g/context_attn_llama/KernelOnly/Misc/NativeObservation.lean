import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Proofs.NativeImplementation

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural NativeBindings
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
open scoped Classical

/-- The numerical premise is exactly the established local instruction law.
All consumers are retained in the observation, including non-owner replicas. -/
theorem native_observes (c : Host.Call) (order : List (GridIndex c.grid))
    (separate : c.regions.start≠c.regions.out ∧ c.regions.sequence≠c.regions.out ∧ c.regions.prompt≠c.regions.out)
    (actual referenceOps : NativeInstructions.ScalarOps)
    (localLaws : NativeInstructions.ScalarLaws actual referenceOps) :
    (nativeIO c order).ObservesOn (observer c order actual) (NativeArrays.metadata c)
      (nativeFormula c order referenceOps) := by
  have observations : (fun values => (nativeIO c order).mapOutputs (observer c order actual)
      (formula c (NativeInstructions.backend c) (NativeInstructions.support c) order values))=
      nativeFormula c order referenceOps := by
    funext values port j
    fin_cases port
    change (fun consumer=>NativeInstructions.evaluate actual
      (NativeInstructions.plan c.layout.BM c.layout.D) _ consumer)=_
    funext consumer
    exact NativeInstructions.evaluated_if_local actual referenceOps localLaws c _ consumer
  rw [←observations]
  exact ArrayProgramIO.ObservesOn.of_implementsOn
    (implements c (NativeInstructions.backend c) (NativeInstructions.support c) order separate)
    (observer c order actual)

#axiomsClean native_observes
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeInterface
