import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Launching

namespace VeriTile.Bench.Repairs.L2NormTriton2.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormTriton1.Host (World Result Keep Dense Frame bind_success context)
open Physical
noncomputable section
open Classical
set_option maxHeartbeats 6000000


def body {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Source.Adapter α γ) (rt : Runtime M) (call : Call γ) (s : World M) : Result M Dense :=
  (prepare rt call s).bind fun q ready =>
  (launch ops a rt q call.eps ready).bind fun _ done => .ok q.output done

theorem body_frame {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (neg : Negation α) (a : Source.Adapter α γ) (rt : Runtime M) (lib : Library ops neg rt)
    (call : Call γ) (s : World M) (domain : Domain call s) :
    Keep ops s (body ops a rt call s).state := by
  have preparation := prepare_frame ops neg rt lib call s domain
  cases hp : prepare rt call s with
  | error reason after => simpa [body,hp,Result.bind,Result.state] using preparation
  | ok q ready =>
    rw [hp] at preparation
    have facts := prepare_sound ops neg rt lib call s domain q ready hp
    have frame := launch_frame ops neg a rt lib q facts.valid call.eps ready
      s.allocations preparation.live facts.outputFresh
    have combined := Frame.trans ops (preparation.frame ops s.allocations (fun _ h => h)) frame.1
    have live : ∀ old∈s.allocations,old∈(launch ops a rt q call.eps ready).state.allocations :=
      fun old h => frame.2 old (preparation.live old h)
    cases hl : launch ops a rt q call.eps ready with
    | error reason after =>
      simpa [body,hp,hl,Result.bind,Result.state] using
        (show Keep ops s (launch ops a rt q call.eps ready).state from ⟨live,combined.reads,combined.bytes⟩)
    | ok value after =>
      simpa [body,hp,hl,Result.bind,Result.state] using
        (show Keep ops s (launch ops a rt q call.eps ready).state from ⟨live,combined.reads,combined.bytes⟩)

/-- All outcomes preserve every initially live allocation, including trailing
bytes outside FP32 words. This includes interrupted launches, exit failures,
and failure of the final metadata reshape; no private write rollback is used. -/
theorem execute_frame {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (neg : Negation α) (a : Source.Adapter α γ) (rt : Runtime M) (lib : Library ops neg rt)
    (call : Call γ) (s : World M) (domain : Domain call s) :
    Keep ops s (execute ops a rt call s).state := by
  by_cases selected : call.selectedFP32=Bool.true
  · have framed := L2NormTriton1.Host.context_frame ops neg rt.forward lib.shared call.device (body ops a rt call) s
      (fun entered keep => body_frame ops neg a rt lib call entered
        ⟨⟨domain.main.backed,keep.live _ domain.main.live,domain.main.device⟩,
          domain.gradientBacked,keep.live _ domain.gradientLive,domain.sameDevice,domain.direct,domain.positive⟩)
    change Keep ops s ((if call.selectedFP32 then
      (context rt.forward call.device (body ops a rt call) s).bind fun y final =>
        match rt.forward.reshape y call.input.shape with
        | some out => .ok out final
        | none => .error "return reshape failed" final
      else .error "backward requires strided FP32 tensors on same CUDA device" s) : Result M View).state
    rw [selected]
    cases hc : context rt.forward call.device (body ops a rt call) s with
    | error reason after => simpa [hc,Result.bind,Result.state] using framed
    | ok y final =>
      cases hr : rt.forward.reshape y call.input.shape <;>
        simpa [hc,hr,Result.bind,Result.state] using framed
  · simpa [execute,selected,Result.state] using Keep.refl ops s

/-- Exact same-call trace. The recorded output is the allocation produced by
prepare, written by this launch, retained by leave, and used by return reshape. -/
def SuccessTrace {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Source.Adapter α γ) (rt : Runtime M) (call : Call γ)
    (s : World M) (out : View) (after : World M) : Prop :=
  call.selectedFP32=Bool.true ∧ ∃ token entered q ready done,
    rt.forward.enter call.device s=.ok token entered ∧
    prepare rt call entered=.ok q ready ∧
    launch ops a rt q call.eps ready=.ok () done ∧
    rt.forward.leave token done=.ok () after ∧
    rt.forward.reshape q.output call.input.shape=some out

theorem execute_trace {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Source.Adapter α γ) (rt : Runtime M) (call : Call γ)
    (s : World M) (out : View) (after : World M)
    (success : execute ops a rt call s=.ok out after) : SuccessTrace ops a rt call s out after := by
  by_cases selected : call.selectedFP32=Bool.true
  · refine ⟨selected,?_⟩
    simp only [execute,selected,if_true] at success
    obtain ⟨y,final,hcontext,hreturn⟩ := bind_success _ _ _ _ success
    cases he : rt.forward.enter call.device s with
    | error reason entered => simp [context,he,Result.bind] at hcontext
    | ok token entered =>
      simp only [context,he,Result.bind] at hcontext
      cases hp : prepare rt call entered with
      | error reason ready =>
        simp only [hp,Result.bind,Result.state] at hcontext
        cases hl : rt.forward.leave token ready <;> simp [hl] at hcontext
      | ok q ready =>
        simp only [hp,Result.bind] at hcontext
        cases hk : launch ops a rt q call.eps ready with
        | error reason done =>
          simp only [hk,Result.bind,Result.state] at hcontext
          cases hl : rt.forward.leave token done <;> simp [hl] at hcontext
        | ok value done =>
          cases value
          simp only [hk,Result.bind,Result.state] at hcontext
          cases hl : rt.forward.leave token done with
          | error reason exit => simp [hl] at hcontext
          | ok value exit =>
            cases value
            simp only [hl,Result.ok.injEq] at hcontext
            obtain ⟨rfl,rfl⟩ := hcontext
            cases hr : rt.forward.reshape q.output call.input.shape with
            | none => simp [hr] at hreturn
            | some returned =>
              simp only [hr,Result.ok.injEq] at hreturn
              obtain ⟨rfl,rfl⟩ := hreturn
              exact ⟨token,entered,q,ready,done,rfl,hp,hk,hl,hr⟩
  · simp [execute,selected] at success

#axiomsClean execute_frame
#axiomsClean execute_trace
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.Host
