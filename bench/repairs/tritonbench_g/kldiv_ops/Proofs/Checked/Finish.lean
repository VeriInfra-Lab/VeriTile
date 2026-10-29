import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Runtime
namespace VeriTile.Bench.Repairs.KldivOps.Contiguous
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open KldivTriton.Host (World Result ForwardCall BackwardCall Arithmetic Fresh Stable Summed Frame Complement
  ScalarReturned Ensures scalarNormalize)
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 6000000

structure ForwardReturned {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (call : ForwardCall) (output : View) (values : Nat → α)
    (out : View) (after : World M) : Prop where
  backed : out.backed
  live : after.live out.storage
  value : if call.mode = .none then out = output ∧
    ∀ i, i < output.numel → logical ops neg after.memory out i = values i
    else ScalarReturned ops neg arith output values
      (KldivTriton.Host.denominator call.mode call.prediction.rows call.prediction.cols) out after

theorem sumAndDivide_spec {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (dim0 : Bool) (input : View) (values : Nat → α) (denom : Option Nat) (s : World M)
    (values_ok : ∀ i, i < input.numel → logical ops neg s.memory input i = values i) :
    Ensures (fun out after => ScalarReturned ops neg arith input values denom out after ∧ Stable ops s after)
      (fun _ after => Stable ops s after) (sumAndDivide rt dim0 input denom s) := by
  unfold sumAndDivide
  have first := lib.sum_stable dim0 input s
  cases e1 : rt.sum dim0 input s with
  | error why failed => simpa only [e1,Result.state] using first
  | ok summed s1 =>
    simp only [e1,Result.state] at first
    simp only [Result.bind]
    obtain ⟨shape,fresh,value⟩ := lib.sum dim0 input s summed s1 e1
    have actual : logical ops neg s1.memory summed.view 0 =
        KldivTriton.HostReduction.eval arith.add arith.zero (fun i => values i.val) summed.plan.tree := by
      rw [value]
      exact KldivTriton.HostReduction.eval_congr _ _ _ _ (fun i => values_ok i.val i.isLt) _
    cases denom with
    | none => exact ⟨⟨shape,fresh.backed,fresh.live,⟨summed.plan,actual⟩⟩,first⟩
    | some n =>
      change Ensures _ _ (rt.base.perform (.divide summed.view n) s1)
      have second := KldivTriton.Host.external_stable ops neg arith rt.base lib.base
        (.divide summed.view n) (by trivial) s1
      cases e2 : rt.base.perform (.divide summed.view n) s1 with
      | error why failed =>
        simp only [e2,Result.state] at second
        exact first.trans second
      | ok out after =>
        simp only [e2,Result.state] at second
        obtain ⟨_,shape2,fresh2,value2⟩ := lib.base.divide summed.view n s1 out after e2
        refine ⟨⟨shape2,fresh2.backed,fresh2.live,⟨summed.plan,?_⟩⟩,first.trans second⟩
        rw [value2,actual]
        rfl

theorem finishForward_spec {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (call : ForwardCall) (output : View) (values : Nat → α) (s : World M)
    (live : s.live output.storage) (backed : output.backed)
    (values_ok : ∀ i, i < output.numel → logical ops neg s.memory output i = values i) :
    Ensures (fun out after => ForwardReturned ops neg arith call output values out after ∧ Stable ops s after)
      (fun _ after => Stable ops s after) (finishForward rt call output s) := by
  by_cases mode : call.mode = .none
  · simp only [finishForward,mode,if_true]
    exact ⟨⟨backed,live,by simpa only [mode,if_true] using And.intro (Eq.refl output) values_ok⟩,Stable.refl ops s⟩
  · have reduced := sumAndDivide_spec ops neg arith rt lib (decide (call.mode = .sum)) output values
      (KldivTriton.Host.denominator call.mode call.prediction.rows call.prediction.cols) s values_ok
    simp only [finishForward,mode,if_false]
    cases e : sumAndDivide rt (decide (call.mode = .sum)) output
        (KldivTriton.Host.denominator call.mode call.prediction.rows call.prediction.cols) s with
    | error why failed => rw [e] at reduced;exact reduced
    | ok out after =>
      rw [e] at reduced
      obtain ⟨spec,stable⟩ := reduced
      exact ⟨⟨spec.backed,spec.live,by simpa only [mode,if_false] using spec⟩,stable⟩

theorem ForwardReturned.memory {M α β : Type} {ops : Ops M α β} {neg : Negation α}
    {arith : Arithmetic α} {call : ForwardCall} {output : View} {values : Nat → α}
    {out : View} {before after : World M} (h : ForwardReturned ops neg arith call output values out before)
    (memory : after.memory = before.memory) (live : after.live out.storage) :
    ForwardReturned ops neg arith call output values out after := by
  refine ⟨h.backed,live,?_⟩
  have value := h.value
  by_cases mode : call.mode = .none
  · simp only [mode,if_true] at value ⊢
    simpa only [memory] using value
  · simp only [mode,if_false] at value ⊢
    exact ⟨value.shape,value.backed,live,by simpa only [memory] using value.value⟩

#axiomsClean finishForward_spec
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Contiguous
