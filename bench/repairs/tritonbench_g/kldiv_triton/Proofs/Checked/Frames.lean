import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Checked.Host
namespace VeriTile.Bench.Repairs.KldivTriton.Contiguous
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open Host
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 6000000

theorem forwardBody_kept {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (adapter : Storage.Adapter α κ)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (call : ForwardCall) (output : Matrix) (s : World M) (kept : List Allocation) (oldLive : ∀ old ∈ kept, s.live old)
    (backed : output.view.backed) (live : s.live output.view.storage)
    (separate : ∀ old ∈ kept, old.separate output.view.storage) :
    Frame ops kept s (forwardBody ops adapter rt call output s).state := by
  let p : Physical.ForwardLayout := ⟨call.prediction,call.target,output,
    blockSize call.prediction.cols,call.flag,modeCode call.mode⟩
  change Frame ops kept s ((rt.perform (.setupForward p s.queue) s).bind _).state
  have fs := external_stable ops neg arith rt lib (.setupForward p s.queue) (by trivial) s
  cases es : rt.perform (.setupForward p s.queue) s with
  | error why failed =>
    simp only [es,Result.state] at fs
    exact fs kept oldLive
  | ok unit before =>
    cases unit
    have same := lib.setupForward _ _ _ _ es
    subst before
    simp only [Result.bind]
    let computed : World M := {s with memory := Storage.forwardKernel ops adapter s.memory p (block_positive _)}
    have fk : Frame ops kept s computed := forward_frame ops adapter p (block_positive _) s kept
      (output.from_backed backed) separate
    have finish := finishForward_stable ops neg arith rt lib call output computed live backed
    exact fk.trans (finish kept oldLive)

theorem runForward_kept {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (adapter : Storage.Adapter α κ)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (a b : Tensor) (call : ForwardCall) (s : World M) (kept : List Allocation) (oldLive : ∀ old ∈ kept, s.live old) :
    Frame ops kept s (runForward ops adapter rt a b call s).state := by
  unfold runForward
  split
  · exact L2NormBwd.Host.Frame.refl _ _ _
  · change Frame ops kept s ((rt.perform (.zeros call.prediction.rows (outputCols call) call.prediction.view.storage.owner) s).bind _).state
    have fz := external_stable ops neg arith rt lib (.zeros call.prediction.rows (outputCols call) call.prediction.view.storage.owner) (by trivial) s
    cases ez : rt.perform (.zeros call.prediction.rows (outputCols call) call.prediction.view.storage.owner) s with
    | error why failed =>
      simp only [ez,Result.state] at fz
      exact fz kept oldLive
    | ok output s1 =>
      simp only [ez,Result.state] at fz
      simp only [Result.bind]
      obtain ⟨_,_,_,_,fresh,_⟩ := lib.zeros _ _ _ _ _ _ ez
      have fe := external_stable ops neg arith rt lib (.enter call.prediction.view.storage.owner) (by trivial) s1
      cases ee : rt.perform (.enter call.prediction.view.storage.owner) s1 with
      | error why failed =>
        simp only [ee,Result.state] at fe
        exact (fz.trans fe) kept oldLive
      | ok token inside =>
        simp only [ee,Result.state] at fe
        simp only [Result.bind]
        have oldLiveInside : ∀ old ∈ kept, inside.live old := fun old ho => (fz.trans fe).live old (oldLive old ho)
        have body := forwardBody_kept ops neg adapter arith rt lib call output inside kept oldLiveInside fresh.backed
          (fe.live _ fresh.live) (fun old ho => separate_symm (fresh.separate old (oldLive old ho)))
        have finish := finallyDevice_frame ops neg arith rt lib token _ inside kept oldLiveInside body
        exact ((fz.trans fe) kept oldLive).trans finish

theorem backwardBody_kept {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (adapter : Storage.Adapter α κ)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (call : BackwardCall) (s : World M) (kept : List Allocation) (oldLive : ∀ old ∈ kept, s.live old) :
    Complement ops call.input.view kept s (backwardBody ops adapter rt call s).state := by
  let p : Physical.BackwardLayout := ⟨call.target,call.input,blockSize call.input.cols,call.flag⟩
  change Complement ops call.input.view kept s ((rt.perform (.setupBackward p s.queue) s).bind _).state
  have fs := external_stable ops neg arith rt lib (.setupBackward p s.queue) (by trivial) s
  cases es : rt.perform (.setupBackward p s.queue) s with
  | error why failed =>
    simp only [es,Result.state] at fs
    exact .ofFrame _ (fs kept oldLive)
  | ok unit before =>
    cases unit
    have same := lib.setupBackward _ _ _ _ es
    subst before
    simp only [Result.bind]
    let computed : World M := {s with memory := Storage.backwardKernel ops adapter s.memory p (block_positive _)}
    have fk := backward_kernel_complement ops adapter p (block_positive _) s kept
    have finish := finishBackward_stable ops neg arith rt lib call call.gradient computed
    exact fk.trans (.ofFrame _ (finish kept oldLive))

theorem runBackward_kept {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (adapter : Storage.Adapter α κ)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (a b g : Tensor) (call : BackwardCall) (s : World M) (kept : List Allocation) (oldLive : ∀ old ∈ kept, s.live old) :
    Complement ops call.input.view kept s (runBackward ops adapter rt a b g call s).state := by
  unfold runBackward
  split
  · exact .ofFrame _ (L2NormBwd.Host.Frame.refl _ _ _)
  · have fe := external_stable ops neg arith rt lib (.enter call.input.view.storage.owner) (by trivial) s
    cases ee : rt.perform (.enter call.input.view.storage.owner) s with
    | error why failed =>
      simp only [ee,Result.state] at fe
      exact .ofFrame _ (fe kept oldLive)
    | ok token inside =>
      simp only [ee,Result.state] at fe
      simp only [Result.bind]
      have oldLiveInside : ∀ old ∈ kept, inside.live old := fun old ho => fe.live old (oldLive old ho)
      have body := backwardBody_kept ops neg adapter arith rt lib call inside kept oldLiveInside
      have finish := finallyDevice_complement ops neg arith rt lib token _ call.input.view inside kept oldLiveInside body
      exact (Complement.ofFrame _ (fe kept oldLive)).trans finish

/-- Public caller frame, specialized from the arbitrary-live-allocation theorem. -/
theorem runForward_frame {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (adapter : Storage.Adapter α κ)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (a b : Tensor) (call : ForwardCall) (s : World M) (domain : ForwardDomain call s) :
    Frame ops call.caller s (runForward ops adapter rt a b call s).state :=
  runForward_kept ops neg adapter arith rt lib a b call s call.caller domain.live

theorem runBackward_complement {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (adapter : Storage.Adapter α κ)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (a b g : Tensor) (call : BackwardCall) (s : World M) (domain : BackwardDomain call s) :
    Complement ops call.input.view call.caller s (runBackward ops adapter rt a b g call s).state :=
  runBackward_kept ops neg adapter arith rt lib a b g call s call.caller domain.live

#axiomsClean runForward_kept
#axiomsClean runBackward_kept
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Contiguous
