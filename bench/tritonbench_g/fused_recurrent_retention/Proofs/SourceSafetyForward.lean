import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedForward
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.SourceSafety
open VeriTile.Triton FP.Structural Quoted
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- These are exact source-address bounds, derived from the public logical
windows below, never numerical assumptions. -/
structure ForwardBounds (s : Shape) (b : InputBuffers) (pid : ProgramId)
    (seed final : Bool) (bounds : RegionName → Nat) : Prop where
  q : ∀ t, t<s.T → ∀ k, k<s.BK → pid.key s k<s.DK → keyOffset (layout s b) pid t k<bounds "q"
  k : ∀ t, t<s.T → ∀ k, k<s.BK → pid.key s k<s.DK → keyOffset (layout s b) pid t k<bounds "k"
  v : ∀ t, t<s.T → ∀ v, v<s.BV → pid.value s v<s.DV → valueOffset (layout s b) pid t v<bounds "v"
  o : ∀ t, t<s.T → ∀ v, v<s.BV → pid.value s v<s.DV → outputOffset (layout s b) pid t v<bounds "o"
  initial : seed=Bool.true → ∀ k, k<s.BK → ∀ v, v<s.BV →
    pid.key s k<s.DK → pid.value s v<s.DV → seedOffset (layout s b) pid k v<bounds "initial_state"
  final : final=Bool.true → ∀ k, k<s.BK → ∀ v, v<s.BV →
    pid.key s k<s.DK → pid.value s v<s.DV → seedOffset (layout s b) pid k v<bounds "final_state"

theorem forward_prefix_safe (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (seed final : Bool) (initial : State (Staged.Node α))
    (hscale : initial.regs .real [] "scale"=some (fun _ => Staged.tagged .scale (a.scale s.DK)))
    (bounds : RegionName → Nat) (hb : ForwardBounds s b (launch initial) seed final bounds) :
    Safety.traceSafeList (Staged.algebra a plans) bounds
      (Program.forwardParts (layout s b) forwardRegions seed final).before initial := by
  cases seed <;>
    simp [Program.forwardParts,Program.forward,Surface.forward,ComputeKernel.surfaceBody,
      Source.dynamicRange?,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,numericLt,
      natLt,integralMod,Region.cast,hscale,layout,forwardRegions,
      Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat]
  constructor
  · intro v k hk hv
    simpa [seedOffset,layout,launch,ProgramId.key,ProgramId.value,Nat.add_assoc] using
      hb.initial rfl k.val k.isLt v.val v.isLt hk hv
  · split <;> trivial

theorem forward_body_safe (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (seed final : Bool) (initial current : State (Staged.Node α)) (pid : ProgramId) (t : Nat)
    (h : ForwardInvariant a s b initial seed pid t current) (ht : t<s.T)
    (bounds : RegionName → Nat) (hb : ForwardBounds s b pid seed final bounds) :
    Safety.traceSafeList (Staged.algebra a plans) bounds
      (Program.forwardParts (layout s b) forwardRegions seed final).body
      (current.setReg "_t" .nat [] (fun _ => t)) := by
  simp [Program.forwardParts,Program.forward,Surface.forward,ComputeKernel.surfaceBody,
    Source.dynamicRange?,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,
    store,Region.cast,layout,forwardRegions,ofFloat,toFloat,
    h.scale,h.keyId,h.valueId,h.headId,h.beta,h.q,h.k,h.v,h.o,h.maskK,h.maskV,h.maskKV,h.carry,
    Broadcast.leftIndex,Broadcast.rightIndex]
  exact ⟨(fun k hk => hb.k t ht k.val k.isLt hk),
    (fun v hv => hb.v t ht v.val v.isLt hv),
    (fun k hk => hb.q t ht k.val k.isLt hk),
    (fun v hv => hb.o t ht v.val v.isLt hv)⟩

theorem forward_final_safe (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (seed final : Bool) (initial current : State (Staged.Node α)) (pid : ProgramId)
    (h : ForwardInvariant a s b initial seed pid s.T current)
    (bounds : RegionName → Nat) (hb : ForwardBounds s b pid seed final bounds) :
    Safety.traceSafeList (Staged.algebra a plans) bounds
      (Program.forwardParts (layout s b) forwardRegions seed final).after current := by
  cases final <;>
    simp [Program.forwardParts,Program.forward,Surface.forward,ComputeKernel.surfaceBody,
      Source.dynamicRange?,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,
      store,Region.cast,layout,forwardRegions,ofFloat,toFloat,
      h.keyId,h.valueId,h.headId,h.maskKV,h.carry,Broadcast.leftIndex,Broadcast.rightIndex]
  constructor
  · intro v k hk hv
    simpa [seedOffset,layout,ProgramId.key,ProgramId.value,Nat.add_assoc] using
      hb.final rfl k.val k.isLt v.val v.isLt hk hv
  · split <;> trivial

end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.SourceSafety
