import bench.tritonbench_g.kldiv_triton.Proofs.Prepare

namespace VeriTile.Bench.Repairs.KldivTriton.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open FP.Equational (ReductionPlan)
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 6000000

def scalarNormalize {α : Type} (arith : Arithmetic α) (denominator : Option Nat) (value : α) : α :=
  match denominator with | none => value | some n => arith.dividePythonNat value n

def sumAndDivide {M : Type} (rt : Runtime M) (input : View) (denominator : Option Nat) (s : World M) : Result M View :=
  (rt.perform (.sum input) s).bind fun summed after =>
  match denominator with | none => .ok summed.view after | some n => rt.perform (.divide summed.view n) after

structure ScalarReturned {M α β : Type} (ops : Ops M α β) (neg : Negation α) (arith : Arithmetic α)
    (input : View) (values : Nat → α) (denominator : Option Nat) (out : View) (after : World M) : Prop where
  shape : out.shape=[]
  backed : out.backed
  live : after.live out.storage
  value : ∃ plan : ReductionPlan input.numel, logical ops neg after.memory out 0=
    scalarNormalize arith denominator (HostReduction.eval arith.add arith.zero (fun i => values i.val) plan.tree)

theorem sumAndDivide_spec {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (input : View) (values : Nat → α) (denominator : Option Nat) (s : World M)
    (values_ok : ∀ i,i < input.numel → logical ops neg s.memory input i=values i) :
    Ensures (fun out after => ScalarReturned ops neg arith input values denominator out after ∧ Stable ops s after)
      (fun _ after => Stable ops s after) (sumAndDivide rt input denominator s) := by
  unfold sumAndDivide
  have first := external_stable ops neg arith rt lib (.sum input) (by trivial) s
  cases e1 : rt.perform (.sum input) s with
  | error why failed => simpa only [e1,Result.state] using first
  | ok summed s1 =>
    simp only [e1,Result.state] at first
    simp only [Result.bind]
    obtain ⟨shape,fresh,value⟩ := lib.sum input s summed s1 e1
    have actual : logical ops neg s1.memory summed.view 0=
        HostReduction.eval arith.add arith.zero (fun i => values i.val) summed.plan.tree := by
      rw [value]
      exact HostReduction.eval_congr _ _ _ _ (fun i => values_ok i.val i.isLt) _
    cases denominator with
    | none => exact ⟨⟨shape,fresh.backed,fresh.live,⟨summed.plan,actual⟩⟩,first⟩
    | some n =>
      change Ensures _ _ (rt.perform (.divide summed.view n) s1)
      have second := external_stable ops neg arith rt lib (.divide summed.view n) (by trivial) s1
      cases e2 : rt.perform (.divide summed.view n) s1 with
      | error why failed =>
        simp only [e2,Result.state] at second
        exact first.trans second
      | ok out after =>
        simp only [e2,Result.state] at second
        obtain ⟨_,shape2,fresh2,value2⟩ := lib.divide summed.view n s1 out after e2
        refine ⟨⟨shape2,fresh2.backed,fresh2.live,⟨summed.plan,?_⟩⟩,first.trans second⟩
        rw [value2,actual]
        rfl

def denominator (mode : Math.Mode) (rows cols : Nat) : Option Nat :=
  match mode with | .none | .sum => none | .mean => some (rows*cols) | .batchmean => some rows

structure ForwardReturned {M α β : Type} (ops : Ops M α β) (neg : Negation α) (arith : Arithmetic α)
    (call : ForwardCall) (output : Matrix) (values : Nat → α) (out : View) (after : World M) : Prop where
  backed : out.backed
  live : after.live out.storage
  value : if call.mode=.none then
    out=output.view ∧ ∀ i,i<output.view.numel → logical ops neg after.memory out i=values i
    else ScalarReturned ops neg arith output.view values
      (denominator call.mode call.prediction.rows call.prediction.cols) out after

theorem finishForward_spec {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (call : ForwardCall) (output : Matrix) (values : Nat → α) (s : World M)
    (live : s.live output.view.storage) (backed : output.view.backed)
    (values_ok : ∀ i,i<output.view.numel → logical ops neg s.memory output.view i=values i) :
    Ensures (fun out after => ForwardReturned ops neg arith call output values out after ∧ Stable ops s after)
      (fun _ after => Stable ops s after) (finishForward rt call output s) := by
  have reduced := sumAndDivide_spec ops neg arith rt lib output.view values
    (denominator call.mode call.prediction.rows call.prediction.cols) s values_ok
  by_cases mode : call.mode=.none
  · simp only [finishForward,mode]
    refine ⟨⟨backed,live,?_⟩,Stable.refl ops s⟩
    simpa only [mode,if_true] using And.intro (Eq.refl output.view) values_ok
  · have run_eq : finishForward rt call output s=
        sumAndDivide rt output.view (denominator call.mode call.prediction.rows call.prediction.cols) s := by
      cases hm : call.mode <;> simp_all [finishForward,sumAndDivide,denominator]
    rw [run_eq]
    cases e : sumAndDivide rt output.view (denominator call.mode call.prediction.rows call.prediction.cols) s with
    | error why failed => rw [e] at reduced; exact reduced
    | ok out after =>
      rw [e] at reduced
      obtain ⟨spec,stable⟩ := reduced
      exact ⟨⟨spec.backed,spec.live,by simpa only [mode,if_false] using spec⟩,stable⟩

/-- Context exit preserves the actual returned object. Success restores the
previous device; body errors also execute exit and retain its failure state. -/
theorem finallyDevice_success {M α β A : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (token : L2NormBwd.Host.Context) (body : Result M A) (value : A) (after : World M)
    (success : finallyDevice rt token body=.ok value after) :
    ∃ before,body=.ok value before ∧ rt.perform (.exit token) before=.ok () after ∧
      after.memory=before.memory ∧ after.device=token.previous := by
  cases body with
  | error why before =>
    unfold finallyDevice at success
    cases e : rt.perform (.exit token) before with
    | error => simp [e,Result.bind] at success
    | ok unit s => cases unit; simp [e,Result.bind] at success
  | ok old before =>
    unfold finallyDevice at success
    cases e : rt.perform (.exit token) before with
    | error => simp [e,Result.bind] at success
    | ok unit s =>
      cases unit
      simp only [e,Result.bind,Result.ok.injEq] at success
      obtain ⟨rfl,rfl⟩ := success
      exact ⟨before,rfl,e,lib.exit token before _ e⟩

#axiomsClean sumAndDivide_spec
#axiomsClean finishForward_spec
#axiomsClean finallyDevice_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Host
