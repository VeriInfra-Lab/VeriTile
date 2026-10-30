import bench.repairs.tritonbench_g.kldiv_triton.Proofs.BackValues
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.ForwardHost

namespace VeriTile.Bench.Repairs.KldivTriton.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 10000000

def BackwardCall.caller (call : BackwardCall) : List Allocation :=
  [call.input.view.storage,call.target.view.storage,call.gradient.storage]

structure BackwardDomain {M : Type} (call : BackwardCall) (initial : World M) : Prop where
  input : call.input.view.backed
  target : call.target.view.backed
  gradient : call.gradient.backed
  live : ∀ old∈call.caller,initial.live old

theorem BackwardReturned.memory {M α β : Type} {ops : Ops M α β} {neg : Negation α} {arith : Arithmetic α}
    {input : Matrix} {shape : List Nat} {iv gv : Nat → α} {out : View} {before after : World M}
    (h : BackwardReturned ops neg arith input shape iv gv out before)
    (memory : after.memory=before.memory) (live : after.live out.storage) :
    BackwardReturned ops neg arith input shape iv gv out after := by
  refine ⟨h.backed,live,?_,?_⟩
  · simpa only [memory] using h.inputValue
  · simpa only [memory] using h.branch

/-- The supplied input is written from the checked source's work buffer. Both
snapshot roles retain pre-call values even if they alias destination words;
the returned product uses the actual broadcast shape, including rank > 2. -/
theorem runBackward_success {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (neg : Negation α)
    (adapter : Storage.Adapter α κ) (arith : Arithmetic α) (rt : Runtime M)
    (lib : Library ops neg arith rt) (call : BackwardCall) (initial after : World M) (out : View)
    (domain : BackwardDomain call initial) (success : runBackward ops adapter rt call initial=.ok out after) :
    BackwardReturned ops neg arith call.input call.gradient.shape (backwardExpected ops neg adapter call initial)
      (fun i => logical ops neg initial.memory call.gradient i) out after ∧ after.device=initial.device := by
  have inputLive : initial.live call.input.view.storage := domain.live _ (by simp [BackwardCall.caller])
  have targetLive : initial.live call.target.view.storage := domain.live _ (by simp [BackwardCall.caller])
  have gradientLive : initial.live call.gradient.storage := domain.live _ (by simp [BackwardCall.caller])
  unfold runBackward at success
  split at success
  · cases success
  next shapes =>
    have rows : call.target.rows=call.input.rows := by by_contra h;exact shapes (Or.inl h)
    have cols : call.target.cols=call.input.cols := by by_contra h;exact shapes (Or.inr h)
    split at success
    · cases success
    next devices =>
      split at success
      · cases success
      next guard =>
        obtain ⟨token,inside,enter,body⟩ := (bind_ok _ _ _ _).mp success
        obtain ⟨returned,hbody,exit,exitMemory,exitDevice⟩ := finallyDevice_success ops neg arith rt lib token _ out after body
        have fe := external_stable ops neg arith rt lib (.enter call.input.view.storage.owner) (by trivial) initial
        simp only [enter,Result.state] at fe
        obtain ⟨previous,_,_⟩ := lib.enter _ _ _ _ enter
        unfold backwardBody at hbody
        obtain ⟨target,s2,et,rest⟩ := (bind_ok _ _ _ _).mp hbody
        obtain ⟨gradient,s4,eg,rest⟩ := (bind_ok _ _ _ _).mp rest
        obtain ⟨work,s5,ew,rest⟩ := (bind_ok _ _ _ _).mp rest
        obtain ⟨u,s6,setup,rest⟩ := (bind_ok _ _ _ _).mp rest
        cases u
        obtain ⟨u,copied,copy,finish⟩ := (bind_ok _ _ _ _).mp rest
        cases u
        have targetSpec := prepareMatrix_spec ops neg arith rt lib call.target Bool.true inside
          (fe.live _ targetLive) domain.target
        rw [et] at targetSpec
        obtain ⟨targetSpec,ft,_⟩ := targetSpec
        have gradientSpec := prepareGradient_spec ops neg arith rt lib call.gradient s2
        rw [eg] at gradientSpec
        obtain ⟨gradientSpec,fg,gradientFresh⟩ := gradientSpec
        have fw := external_stable ops neg arith rt lib (.empty call.input.rows call.input.cols call.input.view.storage.owner) (by trivial) s4
        simp only [ew,Result.state] at fw
        obtain ⟨workRows,workCols,_,unique,fresh⟩ := lib.empty _ _ _ _ _ _ ew
        have beforeStable := ((fe.trans ft).trans fg).trans fw
        have targetValues := ((targetSpec.rebase fe targetLive domain.target).transport fg).transport fw
        have gradientValues := (gradientSpec.rebase (fe.trans ft) gradientLive domain.gradient).transport fw
        have setupSame := lib.setupBackward _ _ _ _ setup
        subst s6
        let p : Physical.BackwardLayout := ⟨target,work,blockSize call.input.cols,call.flag⟩
        let computed : World M := {s5 with memory:=Storage.backwardKernel ops adapter s5.memory p (block_positive _)}
        change copyNoGrad rt call.input.view work.view computed=.ok () copied at copy
        have workSize : work.view.numel=call.input.view.numel := by rw [work.numel,call.input.numel,workRows,workCols]
        have workValues : ∀ i,i<work.view.numel → logical ops neg computed.memory work.view i=backwardExpected ops neg adapter call initial i := by
          intro i hi
          exact backward_kernel_values ops neg adapter call initial s5 target work targetValues
            (workRows.trans rows.symm) (workCols.trans cols.symm) fresh.positive unique i hi
        obtain ⟨_,copiedValues,_⟩ := copyNoGrad_success ops neg arith rt lib call.input.view work.view computed copied copy
        have ivalue (i : Nat) (hi : i<call.input.view.numel) : logical ops neg copied.memory call.input.view i=backwardExpected ops neg adapter call initial i := by
          rw [copiedValues i hi,workValues i (by rw [workSize];exact hi)]
        have gradientFrame : Frame ops [gradient.storage] s5 computed :=
          backward_frame ops adapter p (block_positive _) s5 [gradient.storage]
            (work.from_backed fresh.backed) (by
              intro old ho
              simp only [List.mem_singleton] at ho
              subst old
              exact separate_symm (fresh.separate gradient.storage gradientSpec.live))
        have computedGradientLive : computed.live gradient.storage := fw.live _ gradientSpec.live
        have computedGradient (i : Nat) (hi : i<gradient.numel) : logical ops neg computed.memory gradient i=logical ops neg initial.memory call.gradient i := by
          rw [gradientFrame.logical neg gradient (by simp) gradientSpec.backed i hi]
          exact gradientValues.values i (by simpa [View.numel,gradientSpec.shape] using hi)
        have gradSeparate : gradient.storage.separate call.input.view.storage :=
          gradientFresh _ ((fe.trans ft).live _ inputLive)
        have copyPreserved := copyNoGrad_preserves_success ops neg arith rt lib call.input.view work.view gradient computed copied
          computedGradientLive gradientSpec.backed (fun i j hi hj =>
            gradient.storage.spans_separate call.input.view.storage gradSeparate _ _
              (gradientSpec.backed i hi) (domain.input j hj)) copy
        have gvalue (i : Nat) (hi : i<gradient.numel) : logical ops neg copied.memory gradient i=logical ops neg initial.memory call.gradient i :=
          (copyPreserved.2 i hi).trans (computedGradient i hi)
        have copiedInputLive := copyNoGrad_live_success ops neg arith rt lib call.input.view work.view computed copied copy
          call.input.view.storage (beforeStable.live _ inputLive)
        have finishSpec := finishBackward_success ops neg arith rt lib call gradient
          (backwardExpected ops neg adapter call initial) (fun i => logical ops neg initial.memory call.gradient i)
          copied returned out domain.input copiedInputLive gradientSpec.backed copyPreserved.1 ivalue gvalue finish
        obtain ⟨result,_⟩ := finishSpec
        rw [gradientSpec.shape] at result
        have fx := external_stable ops neg arith rt lib (.exit token) (by trivial) returned
        simp only [exit,Result.state] at fx
        exact ⟨result.memory exitMemory (fx.live _ result.live),exitDevice.trans previous⟩

#axiomsClean runBackward_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Host
