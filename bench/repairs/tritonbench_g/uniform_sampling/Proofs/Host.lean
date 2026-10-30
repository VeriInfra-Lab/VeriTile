import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Execution
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Bounds
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperAction

namespace VeriTile.Bench.Repairs.UniformSampling.Host
open VeriTile.Triton
open VeriTile.Bench.Repairs.LayernormFwdTriton Memory
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical

structure State (Store : Type) where
  memory : Nat→Store
  generators : Nat→Generator
  device : Nat

/-- Runtime/API replies are local. They contain no output values or kernel
success law. Erroring device operations retain their observed device reply. -/
structure Runtime where
  extension : Option Nat
  enter : Option Nat
  allocate : Option Nat
  reserveReturn : Option Nat
  leave : Option Nat
  enterFailureDevice : Nat
  leaveFailureDevice : Nat

structure Stops where
  native : Option Nat
  writeback : Option Nat

def selected (defaults : Nat→Nat) (owner : Nat) (supplied : Option Nat) : Nat :=
  supplied.getD (defaults owner)

def world (api : API α Store Byte) (v : View n) (scratch : Scratch n) :
    World (State Store) (Nat×Nat) Byte where
  observe s key:=api.byte (s.memory key.1) key.2
  caller key:=key.1≠scratch.region ∧ Execution.Outside v key.1 key.2

abbrev A (api : API α Store Byte) (v : View n) (scratch : Scratch n) (X : Type) :=
  Action (world api v scratch) X

def runtime (api : API α Store Byte) (v : View n) (scratch : Scratch n)
    (stage : Stage) (failure : Option Nat) : A api v scratch Unit :=
  call stage {
    run:=fun s=>(s,match failure with |none=>.ok ()|some e=>.error e)
    frame:=by intros;rfl }

def device (api : API α Store Byte) (v : View n) (scratch : Scratch n)
    (target failedDevice : Nat) (failure : Option Nat) : Local (world api v scratch) Unit where
  run s:=({s with device:=if failure.isNone then target else failedDevice},
    match failure with |none=>.ok ()|some e=>.error e)
  frame:=by intros;rfl

def reservation (api : API α Store Byte) (v : View n) (scratch : Scratch n)
    (which : Nat) (returnFailure : Option Nat) : A api v scratch Reservation where
  run s:=match reserve v.device (block n) (counterCount n) (s.generators which) with
    |.error e=>⟨s,.error (.api .admission e),[.admission]⟩
    |.ok (r,g)=>⟨{s with generators:=Function.update s.generators which g},
      match returnFailure with |none=>.ok r|some e=>.error (.api .admission e),[.admission]⟩
  frame s key hk:=by
    unfold world
    split <;> rfl

def native (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (v : View n) (scratch : Scratch n) (r : Reservation) (stop : Option Nat) : A api v scratch Unit where
  run s:=
    let totalLanes:=(IndexProof.schedule n r.start).length
    let count:=match stop with |none=>totalLanes|some k=>min k totalLanes
    ⟨{s with memory:=Execution.nativeMemory api M scalars r.seed r.start scratch count s.memory},
      match stop with |none=>.ok ()|some e=>.error (.api .execute e),[.execute]⟩
  frame s key hk:=Execution.native_prefix_frame api M scalars r.seed r.start scratch _ s.memory _ _ hk.1

def writeback (api : API α Store Byte) (M : Arithmetic α)
    (v : View n) (scratch : Scratch n) (stop : Option Nat) : A api v scratch Unit where
  run s:=
    let count:=match stop with |none=>n|some k=>min k n
    ⟨{s with memory:=Execution.copyMemory api M v scratch count s.memory},
      match stop with |none=>.ok ()|some e=>.error (.api (.transfer .y) e),[.transfer .y]⟩
  frame s key hk:=Execution.copy_prefix_frame api M v scratch _ s.memory _ _ hk.2

def dispatch (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (v : View n) (scratch : Scratch n) (r : Reservation) (stops : Stops) : A api v scratch Nat :=
  (native api M scalars v scratch r stops.native).bind fun _=>
  (writeback api M v scratch stops.writeback).bind fun _=>Action.pure v.handle

def body (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (v : View n) (scratch : Scratch n) (which : Nat) (lib : Runtime) (stops : Stops) : A api v scratch Nat :=
  (runtime api v scratch (.allocate .y) lib.allocate).bind fun _=>
  (reservation api v scratch which lib.reserveReturn).bind fun r=>
    dispatch api M scalars v scratch r stops

def guarded (api : API α Store Byte) (v : View n) (scratch : Scratch n)
    (lib : Runtime) (body : A api v scratch Nat) : A api v scratch Nat where
  run s:= (inDevice (device api v scratch v.device lib.enterFailureDevice lib.enter)
    (device api v scratch s.device lib.leaveFailureDevice lib.leave) body).run s
  frame s key hk:= (inDevice (device api v scratch v.device lib.enterFailureDevice lib.enter)
    (device api v scratch s.device lib.leaveFailureDevice lib.leave) body).frame s key hk

/-- Profile/type/device/scalar-conversion validation is completed before this
entry. Empty returns before overlap checking, extension, allocation or reserve.
For nonempty valid views, overlap validation has proved `View.separate`.
Ordinary invalid-profile/overlap outcomes do not write caller memory or state. -/
def entry (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (v : View n) (scratch : Scratch n) (which : Nat) (lib : Runtime) (stops : Stops) : A api v scratch Nat :=
  if n=0 then Action.pure v.handle else
  (runtime api v scratch .selectKernel lib.extension).bind fun _=>
    guarded api v scratch lib (body api M scalars v scratch which lib stops)

theorem empty (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (v : View 0) (scratch : Scratch 0) (which : Nat) (lib : Runtime) (stops : Stops) (s : State Store) :
    (entry api M scalars v scratch which lib stops).run s=⟨s,.ok v.handle,[]⟩ := by
  simp [entry,Action.pure]

theorem all_outcomes_frame (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (v : View n) (scratch : Scratch n) (which : Nat) (lib : Runtime) (stops : Stops)
    (s : State Store) (r b : Nat) (fresh:r≠scratch.region) (outside:Execution.Outside v r b) :
    api.byte (((entry api M scalars v scratch which lib stops).run s).state.memory r) b=
      api.byte (s.memory r) b := by
  exact (entry api M scalars v scratch which lib stops).frame s (r,b) ⟨fresh,outside⟩

theorem dispatch_success (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (v : View n) (scratch : Scratch n) (r : Reservation) (stops : Stops) (s : State Store) (out : Nat)
    (ok:((dispatch api M scalars v scratch r stops).run s).result=.ok out) :
    out=v.handle ∧
    ((dispatch api M scalars v scratch r stops).run s).state=
      {s with memory:=Execution.run api M scalars r.seed r.start scratch v s.memory} := by
  cases hn:stops.native <;> cases hc:stops.writeback <;>
    simp [dispatch,native,writeback,Action.bind,Action.pure,hn,hc,Execution.run] at ok ⊢
  exact ok.symm

theorem body_success (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (v : View n) (scratch : Scratch n) (which : Nat) (lib : Runtime) (stops : Stops)
    (s : State Store) (out : Nat) (ok:((body api M scalars v scratch which lib stops).run s).result=.ok out) :
    ∃ r g,reserve v.device (block n) (counterCount n) (s.generators which)=.ok (r,g) ∧
      out=v.handle ∧ ((body api M scalars v scratch which lib stops).run s).state=
        ⟨Execution.run api M scalars r.seed r.start scratch v s.memory,
          Function.update s.generators which g,s.device⟩ := by
  cases ha:lib.allocate with
  |some e=>simp [body,runtime,call,Action.bind,ha,Except.mapError] at ok
  |none=>
    simp only [body,runtime,call,ha,Except.mapError,Action.bind] at ok ⊢
    cases hr:reserve v.device (block n) (counterCount n) (s.generators which) with
    |error e=>simp [reservation,hr] at ok
    |ok pair=>
      rcases pair with ⟨r,g⟩
      cases hp:lib.reserveReturn with
      |some e=>simp [reservation,hr,hp] at ok
      |none=>
        simp only [reservation,hr,hp] at ok ⊢
        have h:=dispatch_success api M scalars v scratch r stops
          ⟨s.memory,Function.update s.generators which g,s.device⟩ out ok
        exact ⟨r,g,rfl,h.1,h.2⟩

theorem entry_success (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (v : View n) (scratch : Scratch n) (which : Nat) (lib : Runtime) (stops : Stops)
    (s : State Store) (out : Nat) (hn:0<n)
    (ok:((entry api M scalars v scratch which lib stops).run s).result=.ok out) :
    ∃ r g,reserve v.device (block n) (counterCount n) (s.generators which)=.ok (r,g) ∧
      out=v.handle ∧ ((entry api M scalars v scratch which lib stops).run s).state=
        ⟨Execution.run api M scalars r.seed r.start scratch v s.memory,
          Function.update s.generators which g,s.device⟩ := by
  cases he:lib.extension <;> cases hi:lib.enter <;> cases hl:lib.leave <;>
    simp [entry,Nat.ne_of_gt hn,runtime,call,guarded,inDevice,device,Action.bind,
      he,hi,hl,Except.mapError] at ok ⊢
  obtain ⟨r,g,hr,ho,hstate⟩:=body_success api M scalars v scratch which lib stops
    ⟨s.memory,s.generators,v.device⟩ out ok
  refine ⟨r,g,hr,ho,?_⟩
  rw [hstate]
  exact ⟨rfl,rfl⟩

theorem returned (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (v : View n) (scratch : Scratch n) (which : Nat) (lib : Runtime) (stops : Stops)
    (s : State Store) (out : Nat) (hn:0<n) (neg:∀x,M.neg32 (M.neg32 x)=x)
    (ok:((entry api M scalars v scratch which lib stops).run s).result=.ok out) :
    ∃ r g,reserve v.device (block n) (counterCount n) (s.generators which)=.ok (r,g) ∧
      out=v.handle ∧
      (∀ i,i<n→logical api M ((entry api M scalars v scratch which lib stops).run s).state.memory v i=
        Native.result M scalars r.seed r.start (block n) i) ∧
      ((entry api M scalars v scratch which lib stops).run s).state.generators=
        Function.update s.generators which g ∧
      ((entry api M scalars v scratch which lib stops).run s).state.device=s.device := by
  obtain ⟨r,g,hr,ho,hstate⟩:=entry_success api M scalars v scratch which lib stops s out hn ok
  have hb:=ReservationProof.block_properties n
  have hstart:r.start=alignedStart (s.generators which).offset (block n):=
    (ReservationProof.success hr).2.2.2.2.2.1
  have hs:block n∣r.start:=by rw [hstart];exact ReservationProof.alignment _ _ hb.2.1
  refine ⟨r,g,hr,ho,?_,?_,?_⟩
  · intro i hi
    rw [hstate]
    exact Execution.logical_readback api M scalars r.seed r.start scratch v s.memory neg hs hi
  · rw [hstate]
  · rw [hstate]

#axiomsClean entry_success
#axiomsClean returned

#axiomsClean all_outcomes_frame
#axiomsClean empty
#axiomsClean dispatch_success
#axiomsClean body_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.Host
