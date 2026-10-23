import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Launching

namespace VeriTile.Bench.Repairs.L2NormTriton1.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Physical
noncomputable section
open Classical
set_option maxHeartbeats 6000000

theorem bind_success {M α β : Type} (r : Result M α) (next : α → World M → Result M β)
    (value : β) (after : World M) (success : r.bind next=.ok value after) :
    ∃ a middle,r=.ok a middle ∧ next a middle=.ok value after := by
  cases r with
  | error reason middle => cases success
  | ok a middle => exact ⟨a,middle,rfl,success⟩

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

theorem context_frame {M α β δ : Type} (ops : Ops M α β) (neg : Negation α)
    (rt : Runtime M) (lib : Library ops neg rt) (device : Nat)
    (f : World M → Result M δ) (s : World M)
    (inside : ∀ entered,Keep ops s entered → Keep ops entered (f entered).state) :
    Keep ops s (context rt device f s).state := by
  cases he : rt.enter device s with
  | error reason after => simpa [context,he,Result.bind,Result.state] using lib.enter device s
  | ok token entered =>
    have enteredKeep : Keep ops s entered := by simpa [he,Result.state] using lib.enter device s
    have bodyKeep := Keep.trans ops enteredKeep (inside entered enteredKeep)
    cases hf : f entered with
    | error reason middle =>
      have current : Keep ops s middle := by simpa only [hf,Result.state] using bodyKeep
      have exitKeep := lib.leave token middle
      cases hl : rt.leave token middle <;>
        simpa only [context,he,Result.bind,hf,Result.state,hl] using Keep.trans ops current exitKeep
    | ok value middle =>
      have current : Keep ops s middle := by simpa only [hf,Result.state] using bodyKeep
      have exitKeep := lib.leave token middle
      cases hl : rt.leave token middle <;>
        simpa only [context,he,Result.bind,hf,Result.state,hl] using Keep.trans ops current exitKeep

/-- All outcomes preserve every initially live allocation, including trailing
bytes outside FP32 words. This includes interrupted launches, exit failures,
and failure of the final metadata reshape; no private write rollback is used. -/
theorem execute_frame {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (neg : Negation α) (a : Source.Adapter α γ) (rt : Runtime M) (lib : Library ops neg rt)
    (call : Call γ) (s : World M) (domain : Domain call s) :
    Keep ops s (execute ops a rt call s).state := by
  by_cases selected : call.selectedFP32=Bool.true
  · have framed := context_frame ops neg rt lib call.device (body ops a rt call) s
      (fun entered keep => body_frame ops neg a rt lib call entered
        ⟨domain.backed,keep.live _ domain.live,domain.device,domain.direct⟩)
    change Keep ops s ((if call.selectedFP32 then
      (context rt call.device (body ops a rt call) s).bind fun y final =>
        match rt.reshape y call.input.shape with
        | some out => .ok out final
        | none => .error "return reshape failed" final
      else .error "selected precision requires strided CUDA FP32" s) : Result M View).state
    rw [selected]
    cases hc : context rt call.device (body ops a rt call) s with
    | error reason after => simpa [hc,Result.bind,Result.state] using framed
    | ok y final =>
      cases hr : rt.reshape y call.input.shape <;>
        simpa [hc,hr,Result.bind,Result.state] using framed
  · simpa [execute,selected,Result.state] using Keep.refl ops s

/-- Exact same-call trace. The recorded output is the allocation produced by
prepare, written by this launch, retained by leave, and used by return reshape. -/
def SuccessTrace {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Source.Adapter α γ) (rt : Runtime M) (call : Call γ)
    (s : World M) (out : View) (after : World M) : Prop :=
  call.selectedFP32=Bool.true ∧ ∃ token entered q ready done,
    rt.enter call.device s=.ok token entered ∧
    prepare rt call entered=.ok q ready ∧
    launch ops a rt q call.eps ready=.ok () done ∧
    rt.leave token done=.ok () after ∧
    rt.reshape q.output call.input.shape=some out

theorem execute_trace {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Source.Adapter α γ) (rt : Runtime M) (call : Call γ)
    (s : World M) (out : View) (after : World M)
    (success : execute ops a rt call s=.ok out after) : SuccessTrace ops a rt call s out after := by
  by_cases selected : call.selectedFP32=Bool.true
  · refine ⟨selected,?_⟩
    simp only [execute,selected,if_true] at success
    obtain ⟨y,final,hcontext,hreturn⟩ := bind_success _ _ _ _ success
    cases he : rt.enter call.device s with
    | error reason entered => simp [context,he,Result.bind] at hcontext
    | ok token entered =>
      simp only [context,he,Result.bind] at hcontext
      cases hp : prepare rt call entered with
      | error reason ready =>
        simp only [hp,Result.bind,Result.state] at hcontext
        cases hl : rt.leave token ready <;> simp [hl] at hcontext
      | ok q ready =>
        simp only [hp,Result.bind] at hcontext
        cases hk : launch ops a rt q call.eps ready with
        | error reason done =>
          simp only [hk,Result.bind,Result.state] at hcontext
          cases hl : rt.leave token done <;> simp [hl] at hcontext
        | ok value done =>
          cases value
          simp only [hk,Result.bind,Result.state] at hcontext
          cases hl : rt.leave token done with
          | error reason exit => simp [hl] at hcontext
          | ok value exit =>
            cases value
            simp only [hl,Result.ok.injEq] at hcontext
            obtain ⟨rfl,rfl⟩ := hcontext
            cases hr : rt.reshape q.output call.input.shape with
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
end VeriTile.Bench.Repairs.L2NormTriton1.Host
