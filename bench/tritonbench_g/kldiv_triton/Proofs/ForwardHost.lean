import bench.tritonbench_g.kldiv_triton.Proofs.HostValues
import bench.tritonbench_g.kldiv_triton.Proofs.HostFrames
import bench.tritonbench_g.kldiv_triton.Proofs.Finish

namespace VeriTile.Bench.Repairs.KldivTriton.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 8000000

def ForwardCall.caller (call : ForwardCall) : List Allocation :=
  [call.prediction.view.storage,call.target.view.storage]

structure ForwardDomain {M : Type} (call : ForwardCall) (initial : World M) : Prop where
  prediction : call.prediction.view.backed
  target : call.target.view.backed
  live : ∀ old∈call.caller,initial.live old

theorem ForwardReturned.memory {M α β : Type} {ops : Ops M α β} {neg : Negation α} {arith : Arithmetic α}
    {call : ForwardCall} {output : Matrix} {values : Nat → α} {out : View} {before after : World M}
    (h : ForwardReturned ops neg arith call output values out before)
    (memory : after.memory=before.memory) (live : after.live out.storage) :
    ForwardReturned ops neg arith call output values out after := by
  refine ⟨h.backed,live,?_⟩
  have value := h.value
  by_cases mode : call.mode=.none
  · simp only [mode,if_true] at value ⊢
    simpa only [memory] using value
  · simp only [mode,if_false] at value ⊢
    exact ⟨value.shape,value.backed,live,by simpa only [memory] using value.value⟩

/-- Successful actual host execution returns the internally computed tensor or
an explicitly scheduled scalar reduction. The source execution itself is
constructed by Storage.forwardKernel, not supplied as a success premise. -/
theorem runForward_success {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (neg : Negation α)
    (adapter : Storage.Adapter α κ) (arith : Arithmetic α) (rt : Runtime M)
    (lib : Library ops neg arith rt) (call : ForwardCall) (initial after : World M) (out : View)
    (domain : ForwardDomain call initial) (success : runForward ops adapter rt call initial=.ok out after) :
    ∃ output : Matrix,output.rows=call.prediction.rows ∧ output.cols=outputCols call ∧
      ForwardReturned ops neg arith call output (forwardExpected ops neg adapter call initial) out after ∧
      Frame ops call.caller initial after ∧ after.device=initial.device := by
  have ypLive : initial.live call.prediction.view.storage := domain.live _ (by simp [ForwardCall.caller])
  have tpLive : initial.live call.target.view.storage := domain.live _ (by simp [ForwardCall.caller])
  unfold runForward at success
  split at success
  · cases success
  next shapes =>
    have rows : call.target.rows=call.prediction.rows := by by_contra h;exact shapes (Or.inl h)
    have cols : call.target.cols=call.prediction.cols := by by_contra h;exact shapes (Or.inr h)
    split at success
    · cases success
    next devices =>
      obtain ⟨output,s1,ez,rest⟩ := (bind_ok _ _ _ _).mp success
      obtain ⟨token,inside,ee,body⟩ := (bind_ok _ _ _ _).mp rest
      obtain ⟨returned,hbody,ex,exitMemory,exitDevice⟩ := finallyDevice_success ops neg arith rt lib token _ out after body
      obtain ⟨outRows,outCols,_,unique,fresh,_⟩ := lib.zeros _ _ _ _ _ _ ez
      change output.cols=outputCols call at outCols
      change rt.perform (.zeros call.prediction.rows (outputCols call) call.prediction.view.storage.owner) initial=.ok output s1 at ez
      have fz := external_stable ops neg arith rt lib (.zeros call.prediction.rows (outputCols call) call.prediction.view.storage.owner) (by trivial) initial
      simp only [ez,Result.state] at fz
      have fe := external_stable ops neg arith rt lib (.enter call.prediction.view.storage.owner) (by trivial) s1
      simp only [ee,Result.state] at fe
      obtain ⟨previous,_,_⟩ := lib.enter _ _ _ _ ee
      have initialStable := fz.trans fe
      unfold forwardBody at hbody
      obtain ⟨y,s2,ey,rest⟩ := (bind_ok _ _ _ _).mp hbody
      obtain ⟨t,s4,et,rest⟩ := (bind_ok _ _ _ _).mp rest
      obtain ⟨u,s5,setup,finish⟩ := (bind_ok _ _ _ _).mp rest
      cases u
      have hy := prepareMatrix_spec ops neg arith rt lib call.prediction Bool.false inside
        (initialStable.live _ ypLive) domain.prediction
      rw [ey] at hy
      obtain ⟨hy,fy,_⟩ := hy
      have ht := prepareMatrix_spec ops neg arith rt lib call.target Bool.false s2
        ((initialStable.trans fy).live _ tpLive) domain.target
      rw [et] at ht
      obtain ⟨ht,ft,_⟩ := ht
      have beforeStable := (initialStable.trans fy).trans ft
      have yValues := (hy.rebase initialStable ypLive domain.prediction).transport ft
      have tValues := ht.rebase (initialStable.trans fy) tpLive domain.target
      have setupSame := lib.setupForward _ _ _ _ setup
      subst s5
      let computed : World M := {s4 with memory:=Storage.forwardKernel ops adapter s4.memory ⟨y,t,output,blockSize call.prediction.cols,call.flag,modeCode call.mode⟩ (block_positive _)}
      have values : ∀ i,i<output.view.numel → logical ops neg computed.memory output.view i=
          forwardExpected ops neg adapter call initial i := by
        intro i hi
        exact forward_kernel_values ops neg adapter call initial s4 y t output yValues tValues rows cols
          outRows outCols fresh.positive unique i hi
      have outputLive : computed.live output.view.storage :=
        (fe.trans (fy.trans ft)).live _ fresh.live
      have finishSpec := finishForward_spec ops neg arith rt lib call output
        (forwardExpected ops neg adapter call initial) computed outputLive fresh.backed values
      change finishForward rt call output computed=.ok out returned at finish
      rw [finish] at finishSpec
      obtain ⟨result,ff⟩ := finishSpec
      have fk : Frame ops call.caller s4 computed := forward_frame ops adapter _ (block_positive _) s4 call.caller
        (output.from_backed fresh.backed) (fun old ho => separate_symm (fresh.separate old (domain.live old ho)))
      have callerBefore : Frame ops call.caller initial returned :=
        ((beforeStable call.caller domain.live).trans fk).trans
          (ff call.caller (fun old ho => beforeStable.live old (domain.live old ho)))
      have fx := external_stable ops neg arith rt lib (.exit token) (by trivial) returned
      simp only [ex,Result.state] at fx
      refine ⟨output,outRows,outCols,result.memory exitMemory (fx.live _ result.live),
        callerBefore.trans (fx call.caller (fun old ho => callerBefore.live old (domain.live old ho))),?_⟩
      have d0 := lib.device (.zeros call.prediction.rows (outputCols call) call.prediction.view.storage.owner) initial
      simp only [ez,Result.state] at d0
      exact exitDevice.trans (previous.trans d0)

#axiomsClean runForward_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Host
