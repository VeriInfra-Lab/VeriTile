import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedBackward
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.SourceSafety
open VeriTile.Triton FP.Structural Quoted
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 22000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def finalOffset (p : Quoted.Layout) (pid : ProgramId) (k v : Nat) : Nat :=
  (pid.head/p.H)*p.dfBatch+(pid.head%p.H)*p.dfHead+
    (pid.ik*p.BK+k)*p.dfKey+(pid.iv*p.BV+v)*p.dfValue

structure BackwardBounds (s : Shape) (b : InputBuffers) (pid : ProgramId)
    (seed final : Bool) (bounds : RegionName → Nat) : Prop where
  k : ∀ t,t<s.T → ∀ k,k<s.BK → pid.key s k<s.DK → keyOffset (layout s b) pid t k<bounds "k"
  v : ∀ t,t<s.T → ∀ v,v<s.BV → pid.value s v<s.DV → valueOffset (layout s b) pid t v<bounds "v"
  dout : ∀ t,t<s.T → ∀ v,v<s.BV → pid.value s v<s.DV → doOffset (layout s b) pid t v<bounds "do"
  dq : ∀ t,t<s.T → ∀ k,k<s.BK → pid.key s k<s.DK → gradientOffset (layout s b) pid t k<bounds "dq"
  rq : ∀ r,r<s.T → ∀ k,k<s.BK → pid.key s k<s.DK → reverseKeyOffset (layout s b) pid r k<bounds "q"
  rk : ∀ r,r<s.T → ∀ k,k<s.BK → pid.key s k<s.DK → reverseKeyOffset (layout s b) pid r k<bounds "k"
  rv : ∀ r,r<s.T → ∀ v,v<s.BV → pid.value s v<s.DV → reverseValueOffset (layout s b) pid r v<bounds "v"
  rdo : ∀ r,r<s.T → ∀ v,v<s.BV → pid.value s v<s.DV → reverseDoOffset (layout s b) pid r v<bounds "do"
  rdk : ∀ r,r<s.T → ∀ k,k<s.BK → pid.key s k<s.DK → reverseGradientOffset (layout s b) pid r k<bounds "dk"
  rdv : ∀ r,r<s.T → ∀ v,v<s.BV → pid.value s v<s.DV → reverseOutputOffset (layout s b) pid r v<bounds "dv"
  initial : seed=Bool.true → ∀ k,k<s.BK → ∀ v,v<s.BV →
    pid.key s k<s.DK → pid.value s v<s.DV → seedOffset (layout s b) pid k v<bounds "initial_state"
  final : final=Bool.true → ∀ k,k<s.BK → ∀ v,v<s.BV →
    pid.key s k<s.DK → pid.value s v<s.DV → finalOffset (layout s b) pid k v<bounds "d_final_state"

theorem backward_prefix_safe (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (seed final : Bool) (initial : State (Staged.Node α))
    (hscale : initial.regs .real [] "scale"=some (fun _ => Staged.tagged .scale (a.scale s.DK)))
    (bounds : RegionName → Nat) (hb : BackwardBounds s b (launch initial) seed final bounds) :
    Safety.traceSafeList (Staged.algebra a plans) bounds
      (Program.backwardFirst (layout s b) backwardRegions seed final).before initial := by
  cases seed <;>
    simp [Program.backwardFirst,Program.backward,Surface.backward,ComputeKernel.surfaceBody,
      Source.dynamicRange?,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,numericLt,
      natLt,integralMod,integralFloorDiv,Region.cast,hscale,layout,backwardRegions,
      Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat]
  constructor
  · intro k v hk hv
    simpa [seedOffset,layout,launch,ProgramId.key,ProgramId.value,Nat.add_assoc] using
      hb.initial rfl k.val k.isLt v.val v.isLt hk hv
  · split <;> trivial

theorem backward_first_safe (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (seed final : Bool) (initial current : State (Staged.Node α)) (pid : ProgramId) (t : Nat)
    (h : BackwardFirstInvariant a s b initial seed pid t current) (ht : t<s.T)
    (bounds : RegionName → Nat) (hb : BackwardBounds s b pid seed final bounds) :
    Safety.traceSafeList (Staged.algebra a plans) bounds
      (Program.backwardFirst (layout s b) backwardRegions seed final).body
      (current.setReg "i" .nat [] (fun _ => t)) := by
  simp [Program.backwardFirst,Program.backward,Surface.backward,ComputeKernel.surfaceBody,
    Source.dynamicRange?,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,
    store,Region.cast,layout,backwardRegions,ofFloat,toFloat,
    h.scale,h.keyId,h.valueId,h.headId,h.localHead,h.beta,h.k,h.v,h.dout,h.dq,h.maskK,h.maskV,h.carry,
    Broadcast.leftIndex,Broadcast.rightIndex]
  exact ⟨(fun k hk => hb.k t ht k.val k.isLt hk),
    (fun v hv => hb.v t ht v.val v.isLt hv),
    (fun v hv => hb.dout t ht v.val v.isLt hv),
    (fun k hk => hb.dq t ht k.val k.isLt hk)⟩

theorem backward_middle_safe (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (seed final : Bool) (initial current : State (Staged.Node α)) (pid : ProgramId)
    (h : BackwardFirstInvariant a s b initial seed pid s.T current)
    (bounds : RegionName → Nat) (hb : BackwardBounds s b pid seed final bounds) :
    Safety.traceSafeList (Staged.algebra a plans) bounds
      (Program.backwardSecond (layout s b) backwardRegions seed final).before current := by
  cases final <;>
    simp [Program.backwardSecond,Program.backwardFirst,Program.backward,Surface.backward,ComputeKernel.surfaceBody,
      Source.dynamicRange?,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,
      store,Region.cast,layout,backwardRegions,ofFloat,toFloat,integralFloorDiv,integralMod,
      h.scale,h.keyId,h.valueId,h.headId,h.localHead,h.beta,h.maskK,h.maskV,
      Broadcast.leftIndex,Broadcast.rightIndex]
  constructor
  · intro k v hk hv
    simpa [finalOffset,layout,ProgramId.key,ProgramId.value,Nat.add_assoc] using
      hb.final rfl k.val k.isLt v.val v.isLt hk hv
  · split <;> trivial

theorem backward_second_safe (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (seed final : Bool) (initial current : State (Staged.Node α)) (pid : ProgramId) (r : Nat)
    (h : BackwardSecondInvariant a s b initial final pid r current) (hr : r<s.T)
    (bounds : RegionName → Nat) (hb : BackwardBounds s b pid seed final bounds) :
    Safety.traceSafeList (Staged.algebra a plans) bounds
      (Program.backwardSecond (layout s b) backwardRegions seed final).body
      (current.setReg "_t" .nat [] (fun _ => r)) := by
  simp [Program.backwardSecond,Program.backwardFirst,Program.backward,Surface.backward,ComputeKernel.surfaceBody,
    Source.dynamicRange?,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,
    store,Region.cast,layout,backwardRegions,ofFloat,toFloat,
    h.scale,h.beta,h.q,h.k,h.v,h.dout,h.dk,h.dv,h.maskK,h.maskV,h.carry,
    Broadcast.leftIndex,Broadcast.rightIndex]
  exact ⟨(fun v hv => hb.rdo r hr v.val v.isLt hv),
    (fun k hk => hb.rq r hr k.val k.isLt hk),
    (fun k hk => hb.rk r hr k.val k.isLt hk),
    (fun v hv => hb.rv r hr v.val v.isLt hv),
    (fun k hk => hb.rdk r hr k.val k.isLt hk),
    (fun v hv => hb.rdv r hr v.val v.isLt hv)⟩

end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.SourceSafety
