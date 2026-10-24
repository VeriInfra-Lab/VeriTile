import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Host
namespace VeriTile.Bench.Repairs.KldivOps.Contiguous
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open KldivTriton.Host (World Result ForwardCall BackwardCall Arithmetic Stable Frame Complement external_stable)
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 6000000

theorem finishForward_stable {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (call : ForwardCall) (output : View) (s : World M) (live : s.live output.storage) (backed : output.backed) :
    Stable ops s (finishForward rt call output s).state := by
  have h := finishForward_spec ops neg arith rt lib call output (fun i => logical ops neg s.memory output i)
    s live backed (by intros;rfl)
  cases e : finishForward rt call output s with
  | error => rw [e] at h;exact h
  | ok => rw [e] at h;exact h.2

theorem forwardBody_kept {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (adapter : Adapter α κ)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (call : ForwardCall) (output : View) (eps : α) (s : World M)
    (kept : List Allocation) (oldLive : ∀ old ∈ kept, s.live old)
    (backed : output.backed) (live : s.live output.storage)
    (separate : ∀ old ∈ kept, old.separate output.storage) :
    Frame ops kept s (forwardBody ops adapter rt call output eps s).state := by
  unfold forwardBody
  have fs := lib.forward_stable call output s
  cases es : rt.setupForward call output s with
  | error why failed =>
    simp only [es,Result.state] at fs
    exact fs kept oldLive
  | ok unit before =>
    cases unit
    have same := lib.setupForward _ _ _ _ es
    subst before
    simp only [Result.bind]
    let computed : World M := {s with memory := forwardKernel ops adapter s.memory call.prediction.view call.target.view output call.prediction.rows call.prediction.cols call.flag (reduction call.mode) eps}
    have fk : Frame ops kept s computed := writeView_frame ops output _ s kept backed (Or.inr separate)
    have finish := finishForward_stable ops neg arith rt lib call output computed live backed
    exact fk.trans (finish kept oldLive)

/-- Entire bytes of arbitrary initially live allocations, including every
ordinary failure prefix; allocation and kernel outputs are fresh. -/
theorem runForward_kept {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (adapter : Adapter α κ)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (p t : Tensor) (call : ForwardCall) (eps : α) (s : World M)
    (kept : List Allocation) (oldLive : ∀ old ∈ kept, s.live old) :
    Frame ops kept s (runForward ops adapter rt p t call eps s).state := by
  unfold runForward
  split
  · exact L2NormBwd.Host.Frame.refl _ _ _
  · have fz := lib.zeros_stable (outputShape call) call.prediction.view.storage.owner s
    cases ez : rt.zeros (outputShape call) call.prediction.view.storage.owner s with
    | error why failed =>
      simp only [ez,Result.state] at fz
      exact fz kept oldLive
    | ok output s1 =>
      simp only [ez,Result.state] at fz
      simp only [Result.bind]
      obtain ⟨_,_,fresh,_,_⟩ := lib.zeros _ _ _ _ _ ez
      have fe := external_stable ops neg arith rt.base lib.base (.enter call.prediction.view.storage.owner) (by trivial) s1
      cases ee : rt.base.perform (.enter call.prediction.view.storage.owner) s1 with
      | error why failed =>
        simp only [ee,Result.state] at fe
        exact (fz.trans fe) kept oldLive
      | ok token inside =>
        simp only [ee,Result.state] at fe
        simp only [Result.bind]
        have insideLive : ∀ old ∈ kept, inside.live old := fun old ho => (fz.trans fe).live old (oldLive old ho)
        have body := forwardBody_kept ops neg adapter arith rt lib call output eps inside kept insideLive fresh.backed
          (fe.live _ fresh.live) (fun old ho => KldivTriton.Host.separate_symm (fresh.separate old (oldLive old ho)))
        have finish := KldivTriton.Host.finallyDevice_frame ops neg arith rt.base lib.base token _ inside kept insideLive body
        exact ((fz.trans fe) kept oldLive).trans finish

theorem backwardBody_kept {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (adapter : Adapter α κ)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (call : BackwardCall) (s : World M) (kept : List Allocation)
    (oldLive : ∀ old ∈ kept, s.live old) :
    Complement ops call.input.view kept s (backwardBody ops adapter rt call s).state := by
  unfold backwardBody
  have fs := lib.backward_stable call s
  cases es : rt.setupBackward call s with
  | error why failed =>
    simp only [es,Result.state] at fs
    exact .ofFrame _ (fs kept oldLive)
  | ok unit before =>
    cases unit
    have same := lib.setupBackward _ _ _ es
    subst before
    simp only [Result.bind]
    let computed : World M := {s with memory := backwardKernel ops adapter s.memory call.target.view call.input.view call.input.rows call.input.cols call.flag}
    have fk : Complement ops call.input.view kept s computed := writeView_complement ops call.input.view _ s kept
    have finish := KldivTriton.Host.finishBackward_stable ops neg arith rt.base lib.base call call.gradient computed
    exact fk.trans (.ofFrame _ (finish kept oldLive))

/-- Actual destination mutation and its complete complement survive all
success/failure prefixes, including scalar-one/product failures after launch. -/
theorem runBackward_kept {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (adapter : Adapter α κ)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (d t g : Tensor) (call : BackwardCall) (s : World M)
    (kept : List Allocation) (oldLive : ∀ old ∈ kept, s.live old) :
    Complement ops call.input.view kept s (runBackward ops adapter rt d t g call s).state := by
  unfold runBackward
  split
  · exact .ofFrame _ (L2NormBwd.Host.Frame.refl _ _ _)
  · have fe := external_stable ops neg arith rt.base lib.base (.enter call.input.view.storage.owner) (by trivial) s
    cases ee : rt.base.perform (.enter call.input.view.storage.owner) s with
    | error why failed =>
      simp only [ee,Result.state] at fe
      exact .ofFrame _ (fe kept oldLive)
    | ok token inside =>
      simp only [ee,Result.state] at fe
      simp only [Result.bind]
      have insideLive : ∀ old ∈ kept, inside.live old := fun old ho => fe.live old (oldLive old ho)
      have body := backwardBody_kept ops neg adapter arith rt lib call inside kept insideLive
      have finish := KldivTriton.Host.finallyDevice_complement ops neg arith rt.base lib.base token _ call.input.view inside kept insideLive body
      exact (Complement.ofFrame _ (fe kept oldLive)).trans finish

#axiomsClean runForward_kept
#axiomsClean runBackward_kept
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Contiguous
