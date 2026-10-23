import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Body

namespace VeriTile.Bench.Repairs.L2NormBwd.Host
open VeriTile.Triton FP.Structural Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

theorem normalized_memory {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (call : Call) (initial before after : World M) (q : Request)
    (norm : Normalized ops neg call initial q before) (memory : after.memory=before.memory)
    (frame : Frame ops call.caller before after) : Normalized ops neg call initial q after := by
  refine ⟨norm.shape,norm.cols,norm.positive,norm.block,norm.blockBound,norm.blockActual,
    norm.dyCols,norm.dyRows,norm.dxRows,norm.dxCols,norm.xSize,norm.dySize,norm.xBacked,norm.dyBacked,
    norm.dxBacked,norm.dxUnique,norm.dxPositive,frame.live _ norm.xLive,frame.live _ norm.dyLive,
    frame.live _ norm.dxLive,norm.xFresh,norm.dyFresh,norm.callerFresh,?_,?_,norm.caller.trans frame,
    norm.sameDevice,norm.xMetadata,norm.dyMetadata,norm.dxMetadata,norm.dxDevice⟩
  · intro i hi;rw [memory];exact norm.xValues i hi
  · intro i hi;rw [memory];exact norm.dyValues i hi

/-- Both outcomes execute the real exit operation after an entered context.
Errors retain the actual failing-operation prefix instead of rolling back. -/
theorem contextual_spec {M α β : Type} (ops : Ops M α β) (neg : Negation α) (red : Reduction α)
    (rt : Runtime M) (lib : Library ops neg rt) (call : Call) (initial before : World M)
    (q : Request) (eps : α) (norm : Normalized ops neg call initial q before)
    (domain : InputDomain call initial) :
    Ensures (fun _ after => BodySpec ops neg red call initial q eps after ∧ after.device=before.device)
      (fun _ after => Frame ops call.caller initial after) (contextual ops red rt q eps before) := by
  unfold contextual
  have fe := lib.frame (.enter q.x.view.storage.owner) before call.caller
    (fun a ha => norm.caller.live a (domain.live a ha))
  cases he : rt.perform (.enter q.x.view.storage.owner) before with
  | error why failed =>
    simp only [he,Result.state] at fe
    exact norm.caller.trans fe
  | ok token inside =>
    simp only [he,Result.state] at fe
    simp only [Result.bind]
    obtain ⟨previous,memory,device⟩ := lib.enter _ _ _ _ he
    have normInside := normalized_memory ops neg call initial before inside q norm memory fe
    have spec := body_spec ops neg red rt lib call initial inside q eps normInside domain
    cases hb : body ops red rt q eps inside with
    | error why failed =>
      rw [hb] at spec
      change Frame ops call.caller initial failed at spec
      simp only [finallyExit]
      have fx := lib.frame (.exit token) failed call.caller (fun a ha => spec.live a (domain.live a ha))
      cases hx : rt.perform (.exit token) failed with
      | error reason after =>
        simp only [hx,Result.state] at fx
        exact spec.trans fx
      | ok unit after =>
        cases unit
        simp only [hx,Result.state] at fx
        exact spec.trans fx
    | ok unit computed =>
      cases unit
      rw [hb] at spec
      change BodySpec ops neg red call initial q eps computed at spec
      simp only [finallyExit]
      have fx := lib.frame (.exit token) computed call.caller (fun a ha => spec.caller.live a (domain.live a ha))
      cases hx : rt.perform (.exit token) computed with
      | error reason after =>
        simp only [hx,Result.state] at fx
        exact spec.caller.trans fx
      | ok unit after =>
        cases unit
        simp only [hx,Result.state] at fx
        obtain ⟨kept,restored⟩ := lib.exit _ _ _ hx
        refine ⟨⟨?_,spec.caller.trans fx,fx.live _ spec.dxLive⟩,restored.trans previous⟩
        intro r c hr hc
        rw [kept]
        exact spec.values r c hr hc

theorem prepare_device {M α β : Type} (ops : Ops M α β) (neg : Negation α) (rt : Runtime M)
    (lib : Library ops neg rt) (dy : Matrix) (s : World M) : (prepareDY rt dy s).state.device=s.device := by
  rfl

theorem normalize_device {M α β : Type} (ops : Ops M α β) (neg : Negation α) (rt : Runtime M)
    (lib : Library ops neg rt) (call : Call) (s : World M) : (normalize rt call s).state.device=s.device := by
  unfold normalize
  split
  · rfl
  ·
    have h1 := lib.device (.flatten call.x call.cols) s
    cases e1 : rt.perform (.flatten call.x call.cols) s with
    | error => simpa [e1,Result.bind] using h1
    | ok x s1 =>
      simp only [e1,Result.state] at h1
      simp only [Result.bind]
      have h2 := lib.device (.flatten call.dy call.cols) s1
      cases e2 : rt.perform (.flatten call.dy call.cols) s1 with
      | error => simpa [e2,Result.bind,Result.state] using h2.trans h1
      | ok dy0 s2 =>
        simp only [e2,Result.state] at h2
        simp only [Result.bind]
        have h3 := prepare_device ops neg rt lib dy0 s2
        cases e3 : prepareDY rt dy0 s2 with
        | error => simpa [e3,Result.bind,Result.state] using h3.trans (h2.trans h1)
        | ok dy s3 =>
          simp only [e3,Result.state] at h3
          simp only [Result.bind]
          have h4 := lib.device (.emptyLike x) s3
          cases e4 : rt.perform (.emptyLike x) s3 with
          | error => simpa [e4,Result.bind,Result.state] using h4.trans (h3.trans (h2.trans h1))
          | ok dx s4 =>
            simp only [e4,Result.state] at h4
            simp only [Result.bind]
            split <;> first | exact h4.trans (h3.trans (h2.trans h1)) | skip
            split <;> first | exact h4.trans (h3.trans (h2.trans h1)) | skip
            split <;> exact h4.trans (h3.trans (h2.trans h1))

#axiomsClean contextual_spec
#axiomsClean normalize_device
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Host
