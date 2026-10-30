import bench.tritonbench_g.kldiv_triton.Proofs.BackFinish
import bench.tritonbench_g.kldiv_triton.Proofs.HostFrames

namespace VeriTile.Bench.Repairs.KldivTriton.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 7000000

def backwardExpected {M α β κ : Type} (ops : Ops M α β) (neg : Negation α)
    (adapter : Storage.Adapter α κ) (call : BackwardCall) (initial : World M) (i : Nat) : α :=
  derivative adapter call.flag (logical ops neg initial.memory call.target.view i)

theorem backward_kernel_values {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (neg : Negation α)
    (adapter : Storage.Adapter α κ) (call : BackwardCall) (initial before : World M) (target work : Matrix)
    (ht : PreparedMatrix ops neg call.target target initial before)
    (rows : work.rows=call.target.rows) (cols : work.cols=call.target.cols)
    (positive : work.view.negative=Bool.false) (unique : work.injective)
    (i : Nat) (active : i<work.view.numel) :
    logical ops neg (Storage.backwardKernel ops adapter before.memory
      ⟨target,work,blockSize call.input.cols,call.flag⟩ (block_positive _)) work.view i=
      backwardExpected ops neg adapter call initial i := by
  obtain ⟨hr,hc,index⟩ := flat_bounds work i active
  have keyeq := (congrArg work.view.key index.symm).trans (work.logical_key _ _ hc)
  unfold logical
  simp only [positive,Negation.encode,Bool.false_eq_true,if_false]
  rw [keyeq,Storage.backward_readback ops adapter before.memory _ (block_positive _) unique _ _ hr hc]
  rw [Storage.backward_formula ops adapter before.memory _ (block_positive _) ht.unit _ _ hc
    (by rw [ht.rows,←rows];exact hr) (by rw [ht.cols,←cols];exact hc)]
  change derivative adapter call.flag (Physical.rowValues ops before.memory target (i/work.cols) (i%work.cols))=_
  rw [ht.raw (i/work.cols) (i%work.cols) (by rw [←rows];exact hr) (by rw [←cols];exact hc)]
  rw [←cols,index]
  rfl

theorem copyNoGrad_preserves_success {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (destination source kept : View) (before after : World M)
    (live : before.live kept.storage) (backed : kept.backed)
    (separate : ∀ i j,i<kept.numel → j<destination.numel → (kept.key i).disjoint (destination.key j))
    (success : copyNoGrad rt destination source before=.ok () after) :
    after.live kept.storage ∧ ∀ i,i<kept.numel → logical ops neg after.memory kept i=logical ops neg before.memory kept i := by
  unfold copyNoGrad at success
  obtain ⟨token,inside,enter,body⟩ := (bind_ok _ _ _ _).mp success
  have fe := external_stable ops neg arith rt lib .enterNoGrad (by trivial) before
  simp only [enter,Result.state] at fe
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
    have copiedLive := lib.live (.copyInto destination source) inside kept.storage (fe.live _ live)
    simp only [ec,Result.state] at copiedLive
    have fx := external_stable ops neg arith rt lib (.exitNoGrad token) (by trivial) copied
    simp only [exit,Result.state] at fx
    refine ⟨fx.live _ copiedLive,?_⟩
    intro i hi
    rw [fx.logical neg kept copiedLive backed i hi]
    have copyRead := (lib.copyFrame destination source inside).2 (kept.key i) (fun j hj => separate i j hi hj)
    simp only [ec,Result.state] at copyRead
    unfold logical
    rw [copyRead]
    exact fe.logical neg kept live backed i hi

theorem copyNoGrad_live_success {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (destination source : View) (before after : World M)
    (success : copyNoGrad rt destination source before=.ok () after) :
    ∀ old,before.live old → after.live old := by
  unfold copyNoGrad at success
  obtain ⟨token,inside,enter,body⟩ := (bind_ok _ _ _ _).mp success
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
    intro old ho
    have h1 := lib.live .enterNoGrad before old ho
    simp only [enter,Result.state] at h1
    have h2 := lib.live (.copyInto destination source) inside old h1
    simp only [ec,Result.state] at h2
    have h3 := lib.live (.exitNoGrad token) copied old h2
    simpa only [exit,Result.state] using h3

#axiomsClean backward_kernel_values
#axiomsClean copyNoGrad_preserves_success
#axiomsClean copyNoGrad_live_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Host
