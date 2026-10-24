import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Calls
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
open Numeric

/-- FP32 software value execution with explicit FMA/reduction trees. This
statement does not identify rounded arithmetic with exact real arithmetic. -/
specification reference_fp (u : UnaryModels) (p : PlanBinding s) (call : ForwardCall Word s caller) :
    ForwardReferencePost (arithmetic u) p call (call.sourceRun (arithmetic u) p) ∧
    ∀ (backCaller : Set Nat)
      (back : BackwardCall Word s (call.sourceRun (arithmetic u) p).context call.before call.currentDevice backCaller),
      BackwardReferencePost (arithmetic u) p back (back.sourceRun (arithmetic u) p) := by
  constructor
  · rw [call.sourceRun_eq]
    exact forward_reference (arithmetic u) p call
  · intro backCaller back
    rw [back.sourceRun_eq]
    exact backward_reference (arithmetic u) p back

/-- Whole scalar native composition, conditional only on seven local numerical
instruction correspondences. Compiler graph/layout, host sum instruction
selection, allocator/tensor metadata and physical memory binding are ordinary
structural boundaries; no dot/reduction/whole-output equality is assumed. -/
specification fp_equiv (u : UnaryModels) (ports : ScalarPorts) (laws : ScalarLaws u ports)
    (p : PlanBinding s) (call : ForwardCall Word s caller) :
    call.nativeRun u ports laws p=call.sourceRun (arithmetic u) p ∧
    ForwardReferencePost (arithmetic u) p call (call.nativeRun u ports laws p) ∧
    ∀ (backCaller : Set Nat)
      (back : BackwardCall Word s (call.sourceRun (arithmetic u) p).context call.before call.currentDevice backCaller),
      back.nativeRun u ports laws p=back.sourceRun (arithmetic u) p ∧
      BackwardReferencePost (arithmetic u) p back (back.nativeRun u ports laws p) := by
  refine ⟨call.native_eq u ports laws p,?_,?_⟩
  · rw [call.native_eq u ports laws p,call.sourceRun_eq]
    exact forward_reference (arithmetic u) p call
  · intro backCaller back
    refine ⟨back.native_eq u ports laws p,?_⟩
    rw [back.native_eq u ports laws p,back.sourceRun_eq]
    exact backward_reference (arithmetic u) p back

#axiomsClean reference_fp
#axiomsClean fp_equiv
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
