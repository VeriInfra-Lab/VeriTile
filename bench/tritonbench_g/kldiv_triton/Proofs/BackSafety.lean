import bench.tritonbench_g.kldiv_triton.Proofs.BackwardHost
import bench.tritonbench_g.kldiv_triton.Proofs.ForwardSafety

namespace VeriTile.Bench.Repairs.KldivTriton.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 10000000

structure Complement {M α β : Type} (ops : Ops M α β) (destination : View)
    (kept : List Allocation) (a b : World M) : Prop where
  live : ∀ old,a.live old → b.live old
  reads : ∀ old∈kept,∀ key,old.contains key → (∀ i,i<destination.numel → key.disjoint (destination.key i)) →
    ops.read b.memory key=ops.read a.memory key
  bytes : ∀ old∈kept,∀ owner address,old.covers owner address →
    (∀ i,i<destination.numel → ¬(destination.key i).covers owner address) →
    ops.observe b.memory owner address=ops.observe a.memory owner address
  streams : b.streams=a.streams

theorem Complement.ofFrame {M α β : Type} {ops : Ops M α β} (destination : View)
    {kept : List Allocation} {a b : World M} (frame : Frame ops kept a b) : Complement ops destination kept a b :=
  ⟨frame.live,fun old ho key hk _ => frame.reads old ho key hk,
    fun old ho owner address covered _ => frame.bytes old ho owner address covered,frame.streams⟩

theorem Complement.trans {M α β : Type} {ops : Ops M α β} {destination : View}
    {kept : List Allocation} {a b c : World M} (ab : Complement ops destination kept a b)
    (bc : Complement ops destination kept b c) : Complement ops destination kept a c :=
  ⟨fun old ho => bc.live old (ab.live old ho),
   fun old ho key hk outside => (bc.reads old ho key hk outside).trans (ab.reads old ho key hk outside),
   fun old ho owner address hc outside => (bc.bytes old ho owner address hc outside).trans (ab.bytes old ho owner address hc outside),
   bc.streams.trans ab.streams⟩

theorem finallyGrad_complement {M α β A : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (token : Bool) (body : Result M A) (destination : View) (initial : World M) (kept : List Allocation)
    (live : ∀ old∈kept,initial.live old) (bodyFrame : Complement ops destination kept initial body.state) :
    Complement ops destination kept initial (finallyGrad rt token body).state := by
  cases body with
  | error why s =>
    have f := external_stable ops neg arith rt lib (.exitNoGrad token) (by trivial) s
    cases e : rt.perform (.exitNoGrad token) s with
    | error reason after =>
      simp only [e,Result.state] at f
      simp only [finallyGrad,e,Result.bind,Result.state]
      exact bodyFrame.trans (.ofFrame _ (f kept (fun old ho => bodyFrame.live old (live old ho))))
    | ok unit after =>
      cases unit
      simp only [e,Result.state] at f
      simp only [finallyGrad,e,Result.bind,Result.state]
      exact bodyFrame.trans (.ofFrame _ (f kept (fun old ho => bodyFrame.live old (live old ho))))
  | ok value s =>
    have f := external_stable ops neg arith rt lib (.exitNoGrad token) (by trivial) s
    cases e : rt.perform (.exitNoGrad token) s with
    | error reason after =>
      simp only [e,Result.state] at f
      simp only [finallyGrad,e,Result.bind,Result.state]
      exact bodyFrame.trans (.ofFrame _ (f kept (fun old ho => bodyFrame.live old (live old ho))))
    | ok unit after =>
      cases unit
      simp only [e,Result.state] at f
      simp only [finallyGrad,e,Result.bind,Result.state]
      exact bodyFrame.trans (.ofFrame _ (f kept (fun old ho => bodyFrame.live old (live old ho))))

theorem finallyDevice_complement {M α β A : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (token : L2NormBwd.Host.Context) (body : Result M A) (destination : View) (initial : World M) (kept : List Allocation)
    (live : ∀ old∈kept,initial.live old) (bodyFrame : Complement ops destination kept initial body.state) :
    Complement ops destination kept initial (finallyDevice rt token body).state := by
  cases body with
  | error why s =>
    have f := external_stable ops neg arith rt lib (.exit token) (by trivial) s
    cases e : rt.perform (.exit token) s with
    | error reason after =>
      simp only [e,Result.state] at f
      simp only [finallyDevice,e,Result.bind,Result.state]
      exact bodyFrame.trans (.ofFrame _ (f kept (fun old ho => bodyFrame.live old (live old ho))))
    | ok unit after =>
      cases unit
      simp only [e,Result.state] at f
      simp only [finallyDevice,e,Result.bind,Result.state]
      exact bodyFrame.trans (.ofFrame _ (f kept (fun old ho => bodyFrame.live old (live old ho))))
  | ok value s =>
    have f := external_stable ops neg arith rt lib (.exit token) (by trivial) s
    cases e : rt.perform (.exit token) s with
    | error reason after =>
      simp only [e,Result.state] at f
      simp only [finallyDevice,e,Result.bind,Result.state]
      exact bodyFrame.trans (.ofFrame _ (f kept (fun old ho => bodyFrame.live old (live old ho))))
    | ok unit after =>
      cases unit
      simp only [e,Result.state] at f
      simp only [finallyDevice,e,Result.bind,Result.state]
      exact bodyFrame.trans (.ofFrame _ (f kept (fun old ho => bodyFrame.live old (live old ho))))

theorem copyNoGrad_complement {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (destination source : View) (s : World M) (kept : List Allocation) (live : ∀ old∈kept,s.live old) :
    Complement ops destination kept s (copyNoGrad rt destination source s).state := by
  unfold copyNoGrad
  have fe := external_stable ops neg arith rt lib .enterNoGrad (by trivial) s
  cases e : rt.perform .enterNoGrad s with
  | error why failed =>
    simp only [e,Result.state] at fe
    exact .ofFrame _ (fe kept live)
  | ok token inside =>
    simp only [e,Result.state] at fe
    simp only [Result.bind]
    have copy : Complement ops destination kept inside (rt.perform (.copyInto destination source) inside).state := by
      refine ⟨lib.live _ _,?_,?_,lib.streams _ _⟩
      · intro old ho key hk outside
        exact (lib.copyFrame destination source inside).2 key outside
      · intro old ho owner address hc outside
        exact (lib.copyFrame destination source inside).1 owner address outside
    have final := finallyGrad_complement ops neg arith rt lib token _ destination inside kept
      (fun old ho => fe.live old (live old ho)) copy
    exact (Complement.ofFrame destination (fe kept live)).trans final

theorem finishBackward_stable {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (call : BackwardCall) (gradient : View) (s : World M) : Stable ops s (finishBackward rt call gradient s).state := by
  unfold finishBackward
  have fo := external_stable ops neg arith rt lib (.scalarOne gradient.storage.owner) (by trivial) s
  cases eo : rt.perform (.scalarOne gradient.storage.owner) s with
  | error why failed => simpa only [eo,Result.state] using fo
  | ok one s1 =>
    simp only [eo,Result.state] at fo
    simp only [Result.bind]
    have fe := external_stable ops neg arith rt lib (.equal gradient one) (by trivial) s1
    cases ee : rt.perform (.equal gradient one) s1 with
    | error why failed =>
      simp only [ee,Result.state] at fe
      exact fo.trans fe
    | ok matched s2 =>
      simp only [ee,Result.state] at fe
      simp only [Result.bind]
      split
      · exact fo.trans fe
      · exact (fo.trans fe).trans (external_stable ops neg arith rt lib (.product call.input.view gradient) (by trivial) s2)

theorem prepareGradient_stable {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt) (input : View) (s : World M) :
    Stable ops s (prepareGradient rt input s).state := by
  have h := prepareGradient_spec ops neg arith rt lib input s
  cases e : prepareGradient rt input s with
  | error => rw [e] at h;exact h
  | ok => rw [e] at h;exact h.2.1

theorem backwardBody_complement {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (neg : Negation α)
    (adapter : Storage.Adapter α κ) (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (call : BackwardCall) (s : World M) (domain : BackwardDomain call s) :
    Complement ops call.input.view call.caller s (backwardBody ops adapter rt call s).state := by
  unfold backwardBody
  have ft := prepareMatrix_stable ops neg arith rt lib call.target Bool.true s
    (domain.live _ (by simp [BackwardCall.caller])) domain.target
  cases et : prepareMatrix rt call.target Bool.true s with
  | error why failed =>
    simp only [et,Result.state] at ft
    exact .ofFrame _ (ft call.caller domain.live)
  | ok target s2 =>
    simp only [et,Result.state] at ft
    simp only [Result.bind]
    have fg := prepareGradient_stable ops neg arith rt lib call.gradient s2
    cases eg : prepareGradient rt call.gradient s2 with
    | error why failed =>
      simp only [eg,Result.state] at fg
      exact .ofFrame _ ((ft.trans fg) call.caller domain.live)
    | ok gradient s4 =>
      simp only [eg,Result.state] at fg
      simp only [Result.bind]
      have fw := external_stable ops neg arith rt lib (.empty call.input.rows call.input.cols call.input.view.storage.owner) (by trivial) s4
      cases ew : rt.perform (.empty call.input.rows call.input.cols call.input.view.storage.owner) s4 with
      | error why failed =>
        simp only [ew,Result.state] at fw
        exact .ofFrame _ (((ft.trans fg).trans fw) call.caller domain.live)
      | ok work s5 =>
        simp only [ew,Result.state] at fw
        simp only [Result.bind]
        obtain ⟨_,_,_,_,fresh⟩ := lib.empty _ _ _ _ _ _ ew
        let p : Physical.BackwardLayout := ⟨target,work,blockSize call.input.cols,call.flag⟩
        change Complement ops call.input.view call.caller s ((rt.perform (.setupBackward p s5.queue) s5).bind _).state
        have fs := external_stable ops neg arith rt lib (.setupBackward p s5.queue) (by trivial) s5
        cases es : rt.perform (.setupBackward p s5.queue) s5 with
        | error why failed =>
          simp only [es,Result.state] at fs
          exact .ofFrame _ ((((ft.trans fg).trans fw).trans fs) call.caller domain.live)
        | ok unit s6 =>
          cases unit
          have same := lib.setupBackward _ _ _ _ es
          subst s6
          simp only [Result.bind]
          let computed : World M := {s5 with memory:=Storage.backwardKernel ops adapter s5.memory p (block_positive _)}
          have stable := (ft.trans fg).trans fw
          have fk : Frame ops call.caller s5 computed := backward_frame ops adapter p (block_positive _) s5 call.caller
            (work.from_backed fresh.backed) (fun old ho => separate_symm (fresh.separate old ((ft.trans fg).live old (domain.live old ho))))
          have beforeFrame := (stable call.caller domain.live).trans fk
          have fc := copyNoGrad_complement ops neg arith rt lib call.input.view work.view computed call.caller
            (fun old ho => stable.live old (domain.live old ho))
          change Complement ops call.input.view call.caller s ((copyNoGrad rt call.input.view work.view computed).bind _).state
          cases ec : copyNoGrad rt call.input.view work.view computed with
          | error why failed =>
            simp only [ec,Result.state] at fc
            exact (Complement.ofFrame _ beforeFrame).trans fc
          | ok unit copied =>
            cases unit
            simp only [ec,Result.state] at fc
            simp only [Result.bind]
            have finish := finishBackward_stable ops neg arith rt lib call gradient copied
            exact ((Complement.ofFrame _ beforeFrame).trans fc).trans (.ofFrame _
              (finish call.caller (fun old ho => fc.live old (stable.live old (domain.live old ho)))))

/-- Every ordinary failure prefix preserves the full byte complement of the
supplied view's active words, across all caller allocations and aliases. -/
theorem runBackward_complement {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (neg : Negation α)
    (adapter : Storage.Adapter α κ) (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (call : BackwardCall) (s : World M) (domain : BackwardDomain call s) :
    Complement ops call.input.view call.caller s (runBackward ops adapter rt call s).state := by
  unfold runBackward
  split
  · exact .ofFrame _ (L2NormBwd.Host.Frame.refl _ _ _)
  · split
    · exact .ofFrame _ (L2NormBwd.Host.Frame.refl _ _ _)
    · split
      · exact .ofFrame _ (L2NormBwd.Host.Frame.refl _ _ _)
      · have fe := external_stable ops neg arith rt lib (.enter call.input.view.storage.owner) (by trivial) s
        cases ee : rt.perform (.enter call.input.view.storage.owner) s with
        | error why failed =>
          simp only [ee,Result.state] at fe
          exact .ofFrame _ (fe call.caller domain.live)
        | ok token inside =>
          simp only [ee,Result.state] at fe
          simp only [Result.bind]
          have domainInside : BackwardDomain call inside :=
            ⟨domain.input,domain.target,domain.gradient,fun old ho => fe.live old (domain.live old ho)⟩
          have body := backwardBody_complement ops neg adapter arith rt lib call inside domainInside
          have final := finallyDevice_complement ops neg arith rt lib token _ call.input.view inside call.caller domainInside.live body
          exact (Complement.ofFrame _ (fe call.caller domain.live)).trans final

#axiomsClean runBackward_complement
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Host
