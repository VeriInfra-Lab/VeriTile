import bench.repairs.tritonbench_g.kldiv_triton.Proofs.ForwardHost

namespace VeriTile.Bench.Repairs.KldivTriton.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 8000000

theorem prepareMatrix_stable {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (input : Matrix) (snapshot : Bool) (s : World M) (live : s.live input.view.storage) (backed : input.view.backed) :
    Stable ops s (prepareMatrix rt input snapshot s).state := by
  have h := prepareMatrix_spec ops neg arith rt lib input snapshot s live backed
  cases e : prepareMatrix rt input snapshot s with
  | error => rw [e] at h;exact h
  | ok => rw [e] at h;exact h.2.1

theorem finishForward_stable {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (call : ForwardCall) (output : Matrix) (s : World M) (live : s.live output.view.storage) (backed : output.view.backed) :
    Stable ops s (finishForward rt call output s).state := by
  have h := finishForward_spec ops neg arith rt lib call output (fun i => logical ops neg s.memory output.view i)
    s live backed (by intros;rfl)
  cases e : finishForward rt call output s with
  | error => rw [e] at h;exact h
  | ok => rw [e] at h;exact h.2

theorem finallyDevice_frame {M α β A : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (token : L2NormBwd.Host.Context) (body : Result M A) (initial : World M) (kept : List Allocation)
    (live : ∀ old∈kept,initial.live old) (bodyFrame : Frame ops kept initial body.state) :
    Frame ops kept initial (finallyDevice rt token body).state := by
  cases body with
  | error why s =>
    have f := external_stable ops neg arith rt lib (.exit token) (by trivial) s
    cases e : rt.perform (.exit token) s with
    | error reason after =>
      simp only [e,Result.state] at f
      simp only [finallyDevice,e,Result.bind,Result.state]
      exact bodyFrame.trans (f kept (fun old ho => bodyFrame.live old (live old ho)))
    | ok unit after =>
      cases unit
      simp only [e,Result.state] at f
      simp only [finallyDevice,e,Result.bind,Result.state]
      exact bodyFrame.trans (f kept (fun old ho => bodyFrame.live old (live old ho)))
  | ok value s =>
    have f := external_stable ops neg arith rt lib (.exit token) (by trivial) s
    cases e : rt.perform (.exit token) s with
    | error reason after =>
      simp only [e,Result.state] at f
      simp only [finallyDevice,e,Result.bind,Result.state]
      exact bodyFrame.trans (f kept (fun old ho => bodyFrame.live old (live old ho)))
    | ok unit after =>
      cases unit
      simp only [e,Result.state] at f
      simp only [finallyDevice,e,Result.bind,Result.state]
      exact bodyFrame.trans (f kept (fun old ho => bodyFrame.live old (live old ho)))

theorem forwardBody_frame {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (neg : Negation α)
    (adapter : Storage.Adapter α κ) (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (call : ForwardCall) (output : Matrix) (s : World M) (domain : ForwardDomain call s)
    (backed : output.view.backed) (live : s.live output.view.storage)
    (separate : ∀ old∈call.caller,old.separate output.view.storage) :
    Frame ops call.caller s (forwardBody ops adapter rt call output s).state := by
  unfold forwardBody
  have fy := prepareMatrix_stable ops neg arith rt lib call.prediction Bool.false s
    (domain.live _ (by simp [ForwardCall.caller])) domain.prediction
  cases ey : prepareMatrix rt call.prediction Bool.false s with
  | error why failed =>
    simp only [ey,Result.state] at fy
    exact fy call.caller domain.live
  | ok y s2 =>
    simp only [ey,Result.state] at fy
    simp only [Result.bind]
    have ft := prepareMatrix_stable ops neg arith rt lib call.target Bool.false s2
      (fy.live _ (domain.live _ (by simp [ForwardCall.caller]))) domain.target
    cases et : prepareMatrix rt call.target Bool.false s2 with
    | error why failed =>
      simp only [et,Result.state] at ft
      exact (fy.trans ft) call.caller domain.live
    | ok t s4 =>
      simp only [et,Result.state] at ft
      simp only [Result.bind]
      let p : Physical.ForwardLayout := ⟨y,t,output,blockSize call.prediction.cols,call.flag,modeCode call.mode⟩
      change Frame ops call.caller s ((rt.perform (.setupForward p s4.queue) s4).bind _).state
      have fs := external_stable ops neg arith rt lib (.setupForward p s4.queue) (by trivial) s4
      cases es : rt.perform (.setupForward p s4.queue) s4 with
      | error why failed =>
        simp only [es,Result.state] at fs
        exact ((fy.trans ft).trans fs) call.caller domain.live
      | ok unit s5 =>
        cases unit
        have same := lib.setupForward _ _ _ _ es
        subst s5
        simp only [Result.bind]
        let computed : World M := {s4 with memory:=Storage.forwardKernel ops adapter s4.memory p (block_positive _)}
        have initialFrame := (fy.trans ft) call.caller domain.live
        have fk : Frame ops call.caller s4 computed := forward_frame ops adapter p (block_positive _) s4 call.caller
          (output.from_backed backed) separate
        have finish := finishForward_stable ops neg arith rt lib call output computed
          ((fy.trans ft).live _ live) backed
        exact (initialFrame.trans fk).trans (finish call.caller (fun old ho => (fy.trans ft).live old (domain.live old ho)))

/-- Both successful and failed forward prefixes preserve every byte of every
caller allocation, including unused trailing bytes. Exit failures are ordinary
actual states, not rollback or hidden successful exits. -/
theorem runForward_frame {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (neg : Negation α)
    (adapter : Storage.Adapter α κ) (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (call : ForwardCall) (s : World M) (domain : ForwardDomain call s) :
    Frame ops call.caller s (runForward ops adapter rt call s).state := by
  unfold runForward
  split
  · exact L2NormBwd.Host.Frame.refl _ _ _
  · split
    · exact L2NormBwd.Host.Frame.refl _ _ _
    · change Frame ops call.caller s ((rt.perform (.zeros call.prediction.rows (outputCols call) call.prediction.view.storage.owner) s).bind _).state
      have fz := external_stable ops neg arith rt lib (.zeros call.prediction.rows (outputCols call) call.prediction.view.storage.owner) (by trivial) s
      cases ez : rt.perform (.zeros call.prediction.rows (outputCols call) call.prediction.view.storage.owner) s with
      | error why failed =>
        simp only [ez,Result.state] at fz
        exact fz call.caller domain.live
      | ok output s1 =>
        simp only [ez,Result.state] at fz
        simp only [Result.bind]
        obtain ⟨_,_,_,_,fresh,_⟩ := lib.zeros _ _ _ _ _ _ ez
        have fe := external_stable ops neg arith rt lib (.enter call.prediction.view.storage.owner) (by trivial) s1
        cases ee : rt.perform (.enter call.prediction.view.storage.owner) s1 with
        | error why failed =>
          simp only [ee,Result.state] at fe
          exact (fz.trans fe) call.caller domain.live
        | ok token inside =>
          simp only [ee,Result.state] at fe
          simp only [Result.bind]
          have domainInside : ForwardDomain call inside :=
            ⟨domain.prediction,domain.target,fun old ho => (fz.trans fe).live old (domain.live old ho)⟩
          have body := forwardBody_frame ops neg adapter arith rt lib call output inside domainInside fresh.backed
            (fe.live _ fresh.live) (fun old ho => separate_symm (fresh.separate old (domain.live old ho)))
          have exit := finallyDevice_frame ops neg arith rt lib token _ inside call.caller domainInside.live body
          exact ((fz.trans fe) call.caller domain.live).trans exit

#axiomsClean runForward_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Host
