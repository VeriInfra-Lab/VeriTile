import bench.tritonbench_g.kldiv_triton.Proofs.HostValues
import bench.tritonbench_g.kldiv_triton.Proofs.Finish

namespace VeriTile.Bench.Repairs.KldivTriton.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 7000000

def scalarOneCondition {α : Type} (arith : Arithmetic α) (shape : List Nat) (values : Nat → α) : Prop :=
  shape=[] ∧ arith.equal (values 0) arith.one=Bool.true

structure BackwardReturned {M α β : Type} (ops : Ops M α β) (neg : Negation α) (arith : Arithmetic α)
    (input : Matrix) (gradientShape : List Nat) (inputValues gradientValues : Nat → α)
    (out : View) (after : World M) : Prop where
  backed : out.backed
  live : after.live out.storage
  inputValue : ∀ i,i < input.view.numel → logical ops neg after.memory input.view i=inputValues i
  branch : if scalarOneCondition arith gradientShape gradientValues then out=input.view else
    broadcastShape input.view.shape gradientShape=some out.shape ∧
    ∀ i,i<out.numel → logical ops neg after.memory out i=arith.mul
      (inputValues (broadcastIndex input.view.shape out.shape i))
      (gradientValues (broadcastIndex gradientShape out.shape i))

theorem finishBackward_success {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (call : BackwardCall) (gradient : View) (iv gv : Nat → α) (before after : World M) (out : View)
    (ib : call.input.view.backed) (il : before.live call.input.view.storage)
    (gb : gradient.backed) (gl : before.live gradient.storage)
    (ivalue : ∀ i,i<call.input.view.numel → logical ops neg before.memory call.input.view i=iv i)
    (gvalue : ∀ i,i<gradient.numel → logical ops neg before.memory gradient i=gv i)
    (success : finishBackward rt call gradient before=.ok out after) :
    BackwardReturned ops neg arith call.input gradient.shape iv gv out after ∧ Stable ops before after := by
  unfold finishBackward at success
  obtain ⟨one,s1,eo,rest⟩ := (bind_ok _ _ _ _).mp success
  obtain ⟨matched,s2,ee,finish⟩ := (bind_ok _ _ _ _).mp rest
  have fo := external_stable ops neg arith rt lib (.scalarOne gradient.storage.owner) (by trivial) before
  simp only [eo,Result.state] at fo
  have fe := external_stable ops neg arith rt lib (.equal gradient one) (by trivial) s1
  simp only [ee,Result.state] at fe
  have stable := fo.trans fe
  obtain ⟨oshape,ofresh,ovalue⟩ := lib.scalarOne _ _ _ _ eo
  have gv1 (i : Nat) (hi : i<gradient.numel) : logical ops neg s1.memory gradient i=gv i := by
    rw [fo.logical neg gradient gl gb i hi,gvalue i hi]
  have matchExact : matched=decide (scalarOneCondition arith gradient.shape gv) := by
    rw [lib.equal _ _ _ _ _ ee]
    congr 1
    apply propext
    constructor
    · rintro ⟨shape,equal⟩
      have gs : gradient.shape=[] := shape.trans oshape
      have zero : 0<gradient.numel := by simp [View.numel,gs]
      refine ⟨gs,?_⟩
      have eq0 := equal 0 zero
      rw [gv1 0 zero,ovalue] at eq0
      exact eq0
    · rintro ⟨gs,eq0⟩
      refine ⟨gs.trans oshape.symm,?_⟩
      intro i hi
      have iz : i=0 := by simpa [View.numel,gs] using hi
      subst i
      rw [gv1 0 hi,ovalue]
      exact eq0
  have iv2 (i : Nat) (hi : i<call.input.view.numel) : logical ops neg s2.memory call.input.view i=iv i := by
    rw [stable.logical neg call.input.view il ib i hi,ivalue i hi]
  have gv2 (i : Nat) (hi : i<gradient.numel) : logical ops neg s2.memory gradient i=gv i := by
    rw [stable.logical neg gradient gl gb i hi,gvalue i hi]
  rw [matchExact] at finish
  by_cases scalar : scalarOneCondition arith gradient.shape gv
  · simp only [scalar,decide_true,if_true,Result.ok.injEq] at finish
    obtain ⟨rfl,rfl⟩ := finish
    exact ⟨⟨ib,stable.live _ il,iv2,by simp [scalar]⟩,stable⟩
  · simp only [scalar,decide_false,Bool.false_eq_true,if_false] at finish
    have fp := external_stable ops neg arith rt lib (.product call.input.view gradient) (by trivial) s2
    simp only [finish,Result.state] at fp
    obtain ⟨shape,fresh,values⟩ := lib.product _ _ _ _ _ finish
    refine ⟨⟨fresh.backed,fresh.live,?_,?_⟩,stable.trans fp⟩
    · intro i hi
      rw [fp.logical neg call.input.view (stable.live _ il) ib i hi,iv2 i hi]
    · simp only [scalar,if_false]
      refine ⟨shape,?_⟩
      intro i hi
      obtain ⟨left,right,value⟩ := values i hi
      rw [value,iv2 _ left,gv2 _ right]

/-- Copyback and no_grad exit preserve the actual supplied descriptor. No
assumption says an aliased target/gradient allocation remains unchanged. -/
theorem copyNoGrad_success {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (destination source : View) (before after : World M)
    (success : copyNoGrad rt destination source before=.ok () after) :
    source.numel=destination.numel ∧
      (∀ i,i<destination.numel → logical ops neg after.memory destination i=logical ops neg before.memory source i) ∧
      after.gradEnabled=before.gradEnabled := by
  unfold copyNoGrad at success
  obtain ⟨token,inside,enter,body⟩ := (bind_ok _ _ _ _).mp success
  obtain ⟨previous,enterMemory,_⟩ := lib.enterNoGrad _ _ _ enter
  unfold finallyGrad at body
  cases ec : rt.perform (.copyInto destination source) inside with
  | error why failed =>
    rw [ec] at body
    cases ex : rt.perform (.exitNoGrad token) failed with
    | error => simp [ex,Result.bind] at body
    | ok unit finish => cases unit; simp [ex,Result.bind] at body
  | ok unit copied =>
    cases unit
    rw [ec] at body
    obtain ⟨u,finish,exit,done⟩ := (bind_ok _ _ _ _).mp body
    cases u
    cases done
    obtain ⟨exitMemory,restored⟩ := lib.exitNoGrad _ _ _ exit
    obtain ⟨size,values⟩ := lib.copyInto _ _ _ _ ec
    refine ⟨size,?_,restored.trans previous⟩
    intro i hi
    rw [exitMemory,values i hi,enterMemory]

#axiomsClean finishBackward_success
#axiomsClean copyNoGrad_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Host
