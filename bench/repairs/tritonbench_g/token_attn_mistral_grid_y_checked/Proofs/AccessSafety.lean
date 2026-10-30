import bench.repairs.tritonbench_g.token_attn_mistral.Contract
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.AccessSafety
open VeriTile Triton FP.Structural Safety TokenAttnMistral
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The probability and gathered-value masks differ in the actual source.
Only visited block starts are included, including every padded value lane. -/
structure Bounds (p : Parameters) (s : State α) (bounds : RegionName → Nat) : Prop where
  sequence : s.pids 0*p.sequenceStride<bounds (Region.cast p.length)
  start : s.pids 0*p.attentionStartStride<bounds (Region.cast p.start)
  request : s.pids 0*p.requestStride<bounds (Region.cast p.request)
  count : s.pids 0*p.attentionLengthStride<bounds (Region.cast p.attentionLength)
  probability : 0<p.block → ∀ n,n<TokenAttnMistral.count p s → probOffset p s n<bounds p.prob
  tokens : ∀ n,n<TokenAttnMistral.count p s → n%p.block=0 → ∀ j : Fin p.block,
    n+j.val+windowStart p s<sequenceLength p s → gatherOffset p s (n+j.val)<bounds (Region.cast p.tokens)
  value : ∀ n,n<TokenAttnMistral.count p s → n%p.block=0 → ∀ j : Fin p.block,
    n+j.val+windowStart p s<sequenceLength p s → ∀ d : Fin p.width,
    valueOffset p s d.val+gather p s n j.val*p.valueBatchStride<bounds p.value
  output : ∀ d : Fin p.width,outputOffset p s (d,())<bounds p.output

private theorem natural_window (n : Nat) : (if 0<n then n else 0)=n := by split <;> omega

theorem prelude_safe (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) (inside : Bounds p s bounds) :
    traceSafeList (withSum M R) bounds (prelude p) s := by
  simp (config := {maxSteps:=500000}) [prelude,kernel,source,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,
    numeric,bop,integralFloorDiv,withSum,Region.cast,natLt,numericLt,natural_window,State.setReg]
  exact ⟨inside.sequence,inside.start,inside.request,inside.count⟩

theorem body_safe (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (s t : State α) (n : Nat) (a : TileIndex [p.width] → α)
    (ready : Ready p s t a) (bounds : RegionName → Nat) (inside : Bounds p s bounds)
    (active : n<count p s) (aligned : n%p.block=0) :
    traceSafeList (withSum M R) bounds (body p) (t.setReg "start_n" .nat [] (fun _=>n)) := by
  simp (config := {maxSteps:=500000}) [body,kernel,source,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,
    ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
    numeric,bop,numericLt,natLt,Region.cast,withSum,State.setReg,
    ready.mem,ready.pids,ready.undef,ready.batch,ready.head,ready.offsN,ready.offsD,
    ready.length,ready.sequence,ready.window,ready.prob,ready.gather,ready.value,ready.accumulator]
  refine ⟨?_,?_,?_⟩
  · intro j live
    have h := inside.probability (by have:=j.isLt; omega) (n+j.val) live
    simp only [probOffset,Nat.add_mul] at h ⊢
    convert h using 1 <;> ring
  · intro j live
    have h := inside.tokens n active aligned j live
    simp only [gatherOffset,Nat.add_mul] at h ⊢
    convert h using 1 <;> ring
  · intro j d live
    simpa only [gather,ready.mem,ite_mul,zero_mul] using inside.value n active aligned j live d

theorem loop_safe (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (s t : State α) (n : Nat) (a : TileIndex [p.width] → α)
    (ready : Ready p s t a) (bounds : RegionName → Nat) (inside : Bounds p s bounds)
    (aligned : n%p.block=0) :
    rangeSafe (withSum M R) bounds "start_n" n (count p s) p.block (body p) t := by
  rw [rangeSafe_unfold]
  by_cases zero : p.block=0
  · simp [zero]
  · simp only [zero,if_false]
    by_cases active : n<count p s
    · simp only [active,if_true]
      refine ⟨body_safe M R p s t n a ready bounds inside active aligned,?_⟩
      obtain ⟨next,ran,inv⟩ := body_run M R p s t n a ready
      rw [ran]
      exact loop_safe M R p s next (n+p.block) _ inv bounds inside (by simp [Nat.add_mod,aligned])
    · simp [active]
termination_by count p s-n

theorem postlude_safe (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (s t : State α) (a : TileIndex [p.width] → α) (ready : Ready p s t a)
    (bounds : RegionName → Nat) (inside : Bounds p s bounds) :
    traceSafeList (withSum M R) bounds (postlude p) t := by
  simp (config := {maxSteps:=500000}) [postlude,kernel,source,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
    ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
    numeric,bop,numericLt,natLt,Region.cast,withSum,State.setReg,
    ready.mem,ready.pids,ready.undef,ready.batch,ready.head,ready.offsN,ready.offsD,
    ready.length,ready.sequence,ready.window,ready.prob,ready.gather,ready.value,ready.accumulator]
  exact inside.output

theorem complete (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) (inside : Bounds p s bounds) :
    TraceSafe (withSum M R) bounds (kernel p) s := by
  unfold TraceSafe
  rw [source_split,List.append_assoc,traceSafeList_append]
  refine ⟨prelude_safe M R p s bounds inside,?_⟩
  intro after ran
  obtain ⟨middle,executed,ready⟩ := prelude_run M R p s
  rw [executed] at ran
  injection ran with eq
  subst after
  apply traceSafeList_cons_intro
  · simp only [stepSafe_unfold,safeExpr_unfold,safeOp_unfold,evalOp_unfold,ready.length]
    exact ⟨trivial,trivial,trivial,loop_safe M R p s middle 0 _ ready bounds inside (by simp)⟩
  · intro final ran
    obtain ⟨next,executed,inv⟩ := range_run M R p s middle 0 (count p s) _ ready
    simp only [step,evalOp_unfold,ready.length,Option.bind] at ran
    change FP.Structural.range (withSum M R) "start_n" 0 (count p s) p.block (body p) middle=some final at ran
    rw [executed] at ran
    injection ran with eq
    subst final
    exact postlude_safe M R p s next _ inv bounds inside

#axiomsClean complete
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.AccessSafety
