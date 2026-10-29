import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.Runtime
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.Contract
import bench.repairs.tritonbench_g.layernorm_fwd_triton.revision005.Proofs.Storage

/-! Source-order host-call derivations. A derivation records only the actual
individual host/API results; its final memory and returned tensor are computed
from those calls and the selected instruction interpreter. There is no field
asserting a desired output. The Real interpretation reuses this exact call
below and does not require a second successful wrapper invocation. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton
namespace Wrapper
open Torch Address Memory Execution HostPrepared
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical

/-- Ordinary metadata observations on the actual Tensor handle and data_ptr.
No memory contents or kernel result occurs in this relation. -/
structure MetadataBinding (m : VeriTile.Torch.TensorMetadata) (t : Tensor) : Prop where
  shape : m.shape=t.shape
  dtype : m.dtype = .floating .fp32 ↔ t.dtype=0
  device : m.device.index=t.device
  negative : m.isNeg=t.negative
  pointer : 0<m.numel → m.dataPtr.toNat=t.allocation.base

/-- The successful ordinary Python scalar classification/finiteness query.
It observes the same host scalar packed into the kernel ABI. -/
structure ScalarAPI where
  inspect : Runtime.HostScalar → Contract.Epsilon

structure Request where
  (x w : Tensor)
  eps : Runtime.HostScalar
  (xMetadata wMetadata : VeriTile.Torch.TensorMetadata)
  scalarAPI : ScalarAPI
  xObserved : MetadataBinding xMetadata x
  wObserved : MetadataBinding wMetadata w
  (xStorage wStorage : VeriTile.Bench.Repairs.LayernormFwdTritonRevision005.Storage.Tensor)
  xStorageView : xStorage.view=x
  wStorageView : wStorage.view=w

def Request.epsilon (q : Request) : Contract.Epsilon := q.scalarAPI.inspect q.eps

/-- Exact public admission plus ordinary live-object/current-stream readiness
and complete input-backing observations. Strides and input aliasing need no
additional policy restriction. -/
structure Domain (q : Request) (initial : State Store Host) (caller : Nat → Prop) : Prop where
  admitted : Contract.InputContract q.xMetadata q.wMetadata q.epsilon
  (xPresent : initial.live q.x) (wPresent : initial.live q.w)
  xReady : initial.ready q.x (initial.streams q.x.device)
  wReady : initial.ready q.w (initial.streams q.w.device)
  backing : caller=initial.backing
  xBacking : ∀ i, q.xStorage.backing.base ≤ i →
    i < q.xStorage.backing.base+q.xStorage.backing.nbytes → initial.backing i
  wBacking : ∀ i, q.wStorage.backing.base ≤ i →
    i < q.wStorage.backing.base+q.wStorage.backing.nbytes → initial.backing i

theorem Domain.xFP32 (h : Domain q initial caller) : q.x.dtype=0 :=
  q.xObserved.dtype.mp h.admitted.input.dtype
theorem Domain.wFP32 (h : Domain q initial caller) : q.w.dtype=0 :=
  q.wObserved.dtype.mp h.admitted.weight.dtype
theorem Domain.xPositive (h : Domain q initial caller) : q.x.negative=Bool.false :=
  q.xObserved.negative.symm.trans h.admitted.input.noNeg
theorem Domain.wPositive (h : Domain q initial caller) : q.w.negative=Bool.false :=
  q.wObserved.negative.symm.trans h.admitted.weight.noNeg
theorem Domain.wDevice (h : Domain q initial caller) : q.w.device=q.x.device := by
  rw [←q.wObserved.device,←q.xObserved.device,h.admitted.device]

variable {α Store Byte Host : Type} {api : API α Store Byte} {neg : α → α} {caller : Nat → Prop}

theorem ready_after_effect (effect : Effect api neg caller Host Value) (s : State Store Host)
    (t : Tensor) (h : s.ready t (s.streams t.device)) :
    (effect.call.run s).1.ready t ((effect.call.run s).1.streams t.device) := by
  rw [(effect.preserves s).streams]
  exact (effect.preserves s).ready t _ h

structure Setup (lib : APIHost api neg caller Host) (q : Request) (initial : State Store Host) where
  domain : Domain q initial caller
  shape : Shape
  validated : ((HostPrelude.validate (api:=api) (caller:=caller) q.x q.w).run initial).result=.ok shape
  entered : ((lib.enter q.x.device).call.run initial).2=.ok ()
  preparation : HostPrelude.Route lib q.x q.w ((lib.enter q.x.device).call.run initial).1

variable {lib : APIHost api neg caller Host} {q : Request} {initial : State Store Host}

def Setup.enterState (setup : Setup lib q initial) : State Store Host := ((lib.enter q.x.device).call.run initial).1

def Setup.prepared (setup : Setup lib q initial) : Data setup.shape caller :=
  HostPrepared.ofRoute lib q.x q.w setup.enterState setup.preparation setup.shape
    (HostPrelude.validate_shapes q.x q.w initial setup.shape setup.validated).1
    (HostPrelude.validate_shapes q.x q.w initial setup.shape setup.validated).2
    (q.x.fp32Width setup.domain.xFP32) (q.w.fp32Width setup.domain.wFP32)
    ((lib.enter q.x.device).preserves initial |>.live q.x setup.domain.xPresent)
    ((lib.enter q.x.device).preserves initial |>.live q.w setup.domain.wPresent)
    setup.domain.xPositive setup.domain.wPositive setup.domain.wDevice
    (ready_after_effect (lib.enter q.x.device) initial q.x setup.domain.xReady)
    (ready_after_effect (lib.enter q.x.device) initial q.w setup.domain.wReady)

def Setup.finish (setup : Setup lib q initial) : State Store Host := setup.preparation.finish

def Setup.shadow (setup : Setup lib q initial) : (Nat → ℝ) → (Nat → ℝ) :=
  fun m => setup.preparation.shadow (((lib.enter q.x.device).shadow initial).run m)

theorem setup_ready (setup : Setup lib q initial) : Ready setup.prepared setup.finish :=
  HostPrepared.ofRoute_ready lib q.x q.w setup.enterState setup.preparation setup.shape _ _ _ _ _ _ _ _ _ _ _
    (lib.enter_owner q.x.device initial setup.entered)

theorem setup_inputs (setup : Setup lib q initial) :
    Materialized api neg q.x initial setup.finish setup.shadow setup.prepared.x ∧
    Materialized api neg q.w initial setup.finish setup.shadow setup.prepared.w := by
  obtain ⟨hx,hw⟩ := HostPrelude.route_inputs lib q.x q.w setup.enterState setup.preparation
    ((lib.enter q.x.device).preserves initial |>.live q.x setup.domain.xPresent)
    ((lib.enter q.x.device).preserves initial |>.live q.w setup.domain.wPresent)
    setup.domain.xPositive setup.domain.wPositive
    (ready_after_effect (lib.enter q.x.device) initial q.x setup.domain.xReady)
    (ready_after_effect (lib.enter q.x.device) initial q.w setup.domain.wReady)
  exact ⟨materialized_rebase api neg (lib.enter q.x.device) initial setup.finish _ q.x _ setup.domain.xPresent hx,
    materialized_rebase api neg (lib.enter q.x.device) initial setup.finish _ q.w _ setup.domain.wPresent hw⟩

theorem setup_dtype (setup : Setup lib q initial) :
    setup.prepared.x.dtype=0 ∧ setup.prepared.w.dtype=0 ∧ setup.prepared.y.dtype=0 := by
  have hy := (lib.allocate_dtype q.x setup.enterState setup.preparation.y setup.preparation.allocateY).trans setup.domain.xFP32
  exact ⟨setup.domain.xFP32,setup.domain.wFP32,hy⟩

/-- The single allocation route reconstructs the actual `prepare` result and
post-call state, not merely a tuple with similar values. -/
theorem preparation_execution (r : HostPrelude.Route lib x w start) :
    ((HostPrelude.prepare lib x w).run start).result=.ok (r.xr,r.wp,r.y) ∧
      ((HostPrelude.prepare lib x w).run start).state=r.finish := by
  simp [HostPrelude.prepare,Action.bind,Action.pure,call,Except.mapError,
    r.allocateY,HostPrelude.Route.finish,HostPrelude.Route.xr,HostPrelude.Route.wp]

theorem setup_frame (setup : Setup lib q initial) (i : Nat) (hi : caller i) :
    api.byte setup.finish.memory i=api.byte initial.memory i := by
  have hp := (HostPrelude.prepare lib q.x q.w).frame setup.enterState i hi
  simp only [Setup.enterState] at hp
  rw [(preparation_execution setup.preparation).2] at hp
  exact hp.trans ((lib.enter q.x.device).call.frame initial i hi)

theorem setup_shadow_frame (setup : Setup lib q initial) (m : Nat → ℝ) (i : Nat) (hi : caller i) :
    setup.shadow m i=m i := by
  exact (HostPrelude.route_shadow_frame lib q.x q.w _ setup.preparation _ i hi).trans
    (((lib.enter q.x.device).shadow initial).frame m i hi)

#axiomsClean preparation_execution
#axiomsClean setup_ready
#axiomsClean setup_inputs
#axiomsClean setup_dtype
#axiomsClean setup_frame
#axiomsClean setup_shadow_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end Wrapper
end VeriTile.Bench.Repairs.LayernormFwdTriton
