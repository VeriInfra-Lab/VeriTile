import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.Wrapper

namespace VeriTile.Bench.Repairs.LayernormFwdTriton
namespace Wrapper
open Torch Address Memory Execution HostPrepared
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {α Store Byte Host : Type} {api : API α Store Byte} {neg : α → α} {caller : Nat → Prop}
  {platform : Runtime.Platform α} {service : Runtime.Service platform api neg caller Host}
  {lib : APIHost api neg caller Host} {q : Request} {initial : State Store Host}

def dispatchedState (setup : Setup lib q initial)
    (route : Runtime.Route platform api neg service setup.prepared q.eps setup.finish) : State Store Host :=
  ((Runtime.kernelAction (Host:=Host) platform api setup.prepared q.eps route.code route.schedule).run route.s2).state

/-- A completed source call is a derivation through actual local calls. There
is no output/readback/frame premise and no freely supplied Layout or Plan.
`Setup` derives both actual pointers and strides from the actual prefix. -/
structure SuccessfulCall (platform : Runtime.Platform α) (api : API α Store Byte) (neg : α → α)
    (service : Runtime.Service platform api neg caller Host) (lib : APIHost api neg caller Host)
    (q : Request) (initial : State Store Host) where
  setup : Setup lib q initial
  dispatch : Runtime.Route platform api neg service setup.prepared q.eps setup.finish
  exited : ((lib.leave initial.owner).call.run (dispatchedState setup dispatch)).2=.ok ()

def SuccessfulCall.kernelState (c : SuccessfulCall platform api neg service lib q initial) : State Store Host :=
  dispatchedState c.setup c.dispatch
def SuccessfulCall.final (c : SuccessfulCall platform api neg service lib q initial) : State Store Host :=
  ((lib.leave initial.owner).call.run c.kernelState).1
def SuccessfulCall.returned (c : SuccessfulCall platform api neg service lib q initial) : Tensor := c.setup.prepared.y

def SuccessfulCall.preShadow (c : SuccessfulCall platform api neg service lib q initial) : (Nat → ℝ) → (Nat → ℝ) :=
  fun m => c.dispatch.shadow (c.setup.shadow m)

theorem dispatch_execution (r : Runtime.Route platform api neg service p eps start) :
    ((Runtime.dispatch platform api neg service p eps).run start).result=.ok p.y ∧
    ((Runtime.dispatch platform api neg service p eps).run start).state=
      ((Runtime.kernelAction (Host:=Host) platform api p eps r.code r.schedule).run r.s2).state := by
  simp [Runtime.dispatch,Action.bind,call,Except.mapError,Except.map,Runtime.kernelAction,
    r.compiled,r.admitted,r.completed,Runtime.Route.s2,Runtime.Route.s1]

theorem ready_preserved (p : Data shape caller) (effect : Effect api neg caller Host Value)
    (s : State Store Host) (h : Ready p s) (owner : (effect.call.run s).1.owner=s.owner) :
    Ready p (effect.call.run s).1 := by
  rcases h with ⟨ho,hx,hw,hy,rx,rw,ry⟩
  let hp := effect.preserves s
  refine ⟨owner.trans ho,hp.live p.x hx,hp.live p.w hw,hp.live p.y hy,?_,?_,?_⟩
  · rw [hp.streams];exact hp.ready p.x _ rx
  · rw [hp.streams];exact hp.ready p.w _ rw
  · rw [hp.streams];exact hp.ready p.y _ ry

theorem call_prepared_ready (c : SuccessfulCall platform api neg service lib q initial) :
    Ready c.setup.prepared c.dispatch.s2 := by
  have hc := ready_preserved c.setup.prepared (service.compile c.setup.prepared q.eps) c.setup.finish
    (setup_ready c.setup) (service.compile_owner c.setup.prepared q.eps c.setup.finish)
  exact ready_preserved c.setup.prepared (service.admit c.setup.prepared c.dispatch.code) c.dispatch.s1 hc
    (service.admit_owner c.setup.prepared c.dispatch.code c.dispatch.s1)

theorem call_stream (c : SuccessfulCall platform api neg service lib q initial) :
    c.dispatch.schedule.stream=c.dispatch.s2.streams c.setup.prepared.x.device := by
  have h := service.admitted c.setup.prepared c.dispatch.code c.dispatch.s1 c.dispatch.schedule c.dispatch.admitted
  have ho : c.dispatch.s1.owner=c.setup.prepared.x.device :=
    (service.compile_owner c.setup.prepared q.eps c.setup.finish).trans (setup_ready c.setup).1
  have hs := (service.admit c.setup.prepared c.dispatch.code).preserves c.dispatch.s1 |>.streams
  rw [h.2.2.2,ho]
  exact congrArg (fun streams => streams c.setup.prepared.x.device) hs.symm

theorem call_ready (c : SuccessfulCall platform api neg service lib q initial) :
    c.final.ready c.returned (c.final.streams c.setup.prepared.x.device) := by
  have readyKernel : c.kernelState.ready c.returned (c.kernelState.streams c.setup.prepared.x.device) := by
    right
    exact ⟨c.dispatch.completed,rfl,(call_stream c).symm⟩
  unfold SuccessfulCall.final
  rw [(lib.leave initial.owner).preserves c.kernelState |>.streams]
  exact (lib.leave initial.owner).preserves c.kernelState |>.ready c.returned _ readyKernel

theorem call_restores_owner (c : SuccessfulCall platform api neg service lib q initial) :
    c.final.owner=initial.owner := lib.leave_owner initial.owner c.kernelState c.exited

theorem call_inputs (c : SuccessfulCall platform api neg service lib q initial) :
    Materialized api neg q.x initial c.dispatch.s2 c.preShadow c.setup.prepared.x ∧
    Materialized api neg q.w initial c.dispatch.s2 c.preShadow c.setup.prepared.w := by
  obtain ⟨hx,hw⟩ := setup_inputs c.setup
  have hx1 := materialized_preserved api neg (service.compile c.setup.prepared q.eps) initial c.setup.finish _ q.x _ hx
  have hw1 := materialized_preserved api neg (service.compile c.setup.prepared q.eps) initial c.setup.finish _ q.w _ hw
  exact ⟨materialized_preserved api neg (service.admit c.setup.prepared c.dispatch.code) initial c.dispatch.s1 _ q.x _ hx1,
    materialized_preserved api neg (service.admit c.setup.prepared c.dispatch.code) initial c.dispatch.s1 _ q.w _ hw1⟩

theorem call_kernel_output_live (c : SuccessfulCall platform api neg service lib q initial) :
    c.kernelState.live c.returned := (call_prepared_ready c).2.2.2.1

theorem call_frame (c : SuccessfulCall platform api neg service lib q initial)
    (i : Nat) (hi : initial.backing i) : api.byte c.final.memory i=api.byte initial.memory i := by
  have hc : caller i := by rw [c.setup.domain.backing];exact hi
  have hd := (Runtime.dispatch platform api neg service c.setup.prepared q.eps).frame c.setup.finish i hc
  rw [(dispatch_execution c.dispatch).2] at hd
  exact ((lib.leave initial.owner).call.frame c.kernelState i hc).trans (hd.trans (setup_frame c.setup i hc))

theorem call_pre_shadow_frame (c : SuccessfulCall platform api neg service lib q initial)
    (m : Nat → ℝ) (i : Nat) (hi : caller i) : c.preShadow m i=m i := by
  unfold SuccessfulCall.preShadow Runtime.Route.shadow
  rw [((service.admit c.setup.prepared c.dispatch.code).shadow c.dispatch.s1).frame _ i hi,
    ((service.compile c.setup.prepared q.eps).shadow c.setup.finish).frame _ i hi]
  exact setup_shadow_frame c.setup m i hi

#axiomsClean dispatch_execution
#axiomsClean call_prepared_ready
#axiomsClean call_stream
#axiomsClean call_ready
#axiomsClean call_restores_owner
#axiomsClean call_inputs
#axiomsClean call_frame
#axiomsClean call_pre_shadow_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end Wrapper
end VeriTile.Bench.Repairs.LayernormFwdTriton
