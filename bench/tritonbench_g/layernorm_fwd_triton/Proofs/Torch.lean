import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Grid
import bench.tritonbench_g.layer_norm_fwd.Proofs.WrapperAction

/-! Ordinary local tensor-library contracts. Descriptors are immutable tensor
metadata. Every resolve/copy/allocate event carries its actual completed state,
including on failure. The Real shadow is a per-event coordinate transport on
arbitrary Real memory through the same selected tensor handles; it is not a
second host run and never decodes an FP word. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton
namespace Torch
open Address Memory Execution
open VeriTile.Bench.Repairs.LayerNormFwd.Address (byteAddress)

noncomputable section
open scoped Classical

structure Tensor where
  object : Nat
  (shape strides : List Nat)
  rank : strides.length=shape.length
  allocation : Allocation
  dtype : Nat
  fp32Width : dtype=0 → allocation.width=4
  device : Nat
  negative : Bool
  shapeSigned : ∀ n : Nat, n∈shape → n < (2^63 : Nat)
  strideSigned : ∀ n : Nat, n∈strides → n < (2^63 : Nat)
  bounds : ∀ index : List Nat, List.Forall₂ (fun (i n : Nat) => i < n) index shape →
    (List.zipWith (fun (i n : Nat) => i*n) index strides).sum < allocation.extent

abbrev Coordinate (t : Tensor) (index : List Nat) := List.Forall₂ (fun i n => i<n) index t.shape

def address (t : Tensor) (index : List Nat) : Nat :=
  byteAddress t.allocation (List.zipWith (fun (i n : Nat) => i*n) index t.strides).sum

def logical (api : API α Store Byte) (neg : α → α) (m : Store) (t : Tensor) (index : List Nat) : α :=
  if t.negative then neg (api.read m (address t index)) else api.read m (address t index)
def realLogical (m : Nat → ℝ) (t : Tensor) (index : List Nat) : ℝ :=
  if t.negative then -(m (address t index)) else m (address t index)

def Covers (t : Tensor) (i : Nat) : Prop :=
  t.allocation.base ≤ i ∧ i < t.allocation.base+t.allocation.extent*t.allocation.width

structure State (Store Host : Type) where
  memory : Store
  metadata : Host
  /-- Complete physical caller backing as observed on entry, including gaps and storage outside input views. -/
  backing : Nat → Prop
  live : Tensor → Prop
  owner : Nat
  streams : Nat → Nat
  ready : Tensor → Nat → Prop

def world (api : API α Store Byte) (caller : Nat → Prop) : VeriTile.Bench.Repairs.LayerNormFwd.Wrapper.World (State Store Host) Nat Byte where
  observe s i := api.byte s.memory i
  caller := caller

/-- Ordinary preservation by an individual read-only tensor/metadata call.
Ready means ready on this actual device's actual current stream. The input
readiness obligation is retained explicitly; no cross-stream wait is invented. -/
structure Preserves (api : API α Store Byte) (neg : α → α)
    (before after : State Store Host) : Prop where
  live : ∀ t, before.live t → after.live t
  values : ∀ t, before.live t → ∀ index, Coordinate t index →
    logical api neg after.memory t index=logical api neg before.memory t index
  streams : after.streams=before.streams
  ready : ∀ t stream, before.ready t stream → after.ready t stream

structure Shadow (caller : Nat → Prop) (before : State Store Host) where
  run : (Nat → ℝ) → (Nat → ℝ)
  frame : ∀ m i, caller i → run m i=m i
  values : ∀ m t, before.live t → ∀ index, Coordinate t index →
    realLogical (run m) t index=realLogical m t index

/-- A local effect has a real state on both branches and a local, universally
quantified Real transport. No kernel or output specification occurs here. -/
structure Effect (api : API α Store Byte) (neg : α → α) (caller : Nat → Prop) (Host Value : Type) where
  call : VeriTile.Bench.Repairs.LayerNormFwd.Wrapper.Local (world (Host:=Host) api caller) Value
  preserves : ∀ s, Preserves api neg s (call.run s).1
  shadow : ∀ s : State Store Host, Shadow caller s

structure APIHost (api : API α Store Byte) (neg : α → α) (caller : Nat → Prop) (Host : Type) where
  enter : Nat → Effect api neg caller Host Unit
  enter_owner : ∀ device s, ((enter device).call.run s).2=.ok () →
    ((enter device).call.run s).1.owner=device
  leave : Nat → Effect api neg caller Host Unit
  leave_owner : ∀ device s, ((leave device).call.run s).2=.ok () →
    ((leave device).call.run s).1.owner=device
  resolve : Tensor → Effect api neg caller Host Tensor
  resolve_metadata : ∀ t s u, ((resolve t).call.run s).2=.ok u →
    u.shape=t.shape ∧ u.allocation.width=t.allocation.width ∧
    u.device=t.device ∧ u.negative=Bool.false ∧ ((resolve t).call.run s).1.live u
  resolve_dtype : ∀ t s u, ((resolve t).call.run s).2=.ok u → u.dtype=t.dtype
  resolve_values : ∀ t s u, ((resolve t).call.run s).2=.ok u →
    ∀ index, Coordinate t index →
      logical api neg ((resolve t).call.run s).1.memory u index=logical api neg s.memory t index
  resolve_shadow : ∀ t s u, ((resolve t).call.run s).2=.ok u →
    ∀ m index, Coordinate t index →
      realLogical ((resolve t).shadow s |>.run m) u index=realLogical m t index
  resolve_ready : ∀ t s u, ((resolve t).call.run s).2=.ok u →
    s.ready t (s.streams t.device) →
      ((resolve t).call.run s).1.ready u (s.streams u.device)
  resolve_owner : ∀ t s, ((resolve t).call.run s).1.owner=s.owner
  transfer : Nat → Tensor → Effect api neg caller Host Tensor
  transfer_metadata : ∀ device t s u, ((transfer device t).call.run s).2=.ok u →
    u.shape=t.shape ∧ u.allocation.width=t.allocation.width ∧
    u.device=device ∧ u.negative=t.negative ∧ ((transfer device t).call.run s).1.live u
  transfer_dtype : ∀ device t s u, ((transfer device t).call.run s).2=.ok u → u.dtype=t.dtype
  transfer_values : ∀ device t s u, ((transfer device t).call.run s).2=.ok u →
    ∀ index, Coordinate t index →
      logical api neg ((transfer device t).call.run s).1.memory u index=logical api neg s.memory t index
  transfer_shadow : ∀ device t s u, ((transfer device t).call.run s).2=.ok u →
    ∀ m index, Coordinate t index →
      realLogical ((transfer device t).shadow s |>.run m) u index=realLogical m t index
  transfer_ready : ∀ device t s u, ((transfer device t).call.run s).2=.ok u →
    s.ready t (s.streams t.device) →
      ((transfer device t).call.run s).1.ready u (s.streams device)
  transfer_owner : ∀ device t s, ((transfer device t).call.run s).1.owner=s.owner
  allocate : Tensor → Effect api neg caller Host Tensor
  allocate_metadata : ∀ t s u, ((allocate t).call.run s).2=.ok u →
    u.shape=t.shape ∧ u.allocation.width=t.allocation.width ∧
    u.device=t.device ∧ u.negative=Bool.false ∧ ((allocate t).call.run s).1.live u
  allocate_dtype : ∀ t s u, ((allocate t).call.run s).2=.ok u → u.dtype=t.dtype
  allocate_aligned : ∀ t s u, ((allocate t).call.run s).2=.ok u →
    (∃ index, Coordinate u index) → u.allocation.base % 4=0
  allocate_injective : ∀ t s u, ((allocate t).call.run s).2=.ok u →
    ∀ i j, Coordinate u i → Coordinate u j → address u i=address u j → i=j
  allocate_fresh : ∀ t s u, ((allocate t).call.run s).2=.ok u →
    ∀ i, Covers u i → ¬caller i ∧ ∀ old, s.live old → ¬Covers old i
  allocate_ready : ∀ t s u, ((allocate t).call.run s).2=.ok u →
    ((allocate t).call.run s).1.ready u (s.streams u.device)
  allocate_owner : ∀ t s, ((allocate t).call.run s).1.owner=s.owner

/-- An input materialized by local calls; no result-output fields. -/
structure Materialized (api : API α Store Byte) (neg : α → α)
    (original : Tensor) (initial : State Store Host) (current : State Store Host)
    (shadow : (Nat → ℝ) → (Nat → ℝ)) (tensor : Tensor) : Prop where
  shape : tensor.shape=original.shape
  width : tensor.allocation.width=original.allocation.width
  positive : tensor.negative=Bool.false
  present : current.live tensor
  values : ∀ index, Coordinate original index →
    logical api neg current.memory tensor index=logical api neg initial.memory original index
  realValues : ∀ m index, Coordinate original index →
    realLogical (shadow m) tensor index=realLogical m original index
  ready : current.ready tensor (current.streams tensor.device)

/-- Preservation through one later local call is proved from that call's
ordinary old-object preservation and its Real shadow transport. -/
theorem materialized_preserved (api : API α Store Byte) (neg : α → α)
    (e : Effect api neg caller Host Value) (initial current : State Store Host)
    (shadow : (Nat → ℝ) → (Nat → ℝ)) (original tensor : Tensor)
    (h : Materialized api neg original initial current shadow tensor) :
    Materialized api neg original initial (e.call.run current).1
      (fun m => (e.shadow current).run (shadow m)) tensor := by
  let hp := e.preserves current
  refine ⟨h.shape,h.width,h.positive,hp.live tensor h.present,?_,?_,?_⟩
  · intro index hi
    have ht : Coordinate tensor index := by simpa only [Coordinate,h.shape] using hi
    exact (hp.values tensor h.present index ht).trans (h.values index hi)
  · intro m index hi
    have ht : Coordinate tensor index := by simpa only [Coordinate,h.shape] using hi
    exact ((e.shadow current).values (shadow m) tensor h.present index ht).trans (h.realValues m index hi)
  · rw [hp.streams]
    exact hp.ready tensor _ h.ready

theorem resolve_start (api : API α Store Byte) (neg : α → α)
    (lib : APIHost api neg caller Host) (s : State Store Host) (t u : Tensor)
    (ok : ((lib.resolve t).call.run s).2=.ok u)
    (ready : s.ready t (s.streams t.device)) :
    Materialized api neg t s ((lib.resolve t).call.run s).1
      ((lib.resolve t).shadow s).run u := by
  obtain ⟨hs,hw,hd,hn,hp⟩ := lib.resolve_metadata t s u ok
  refine ⟨hs,hw,hn,hp,lib.resolve_values t s u ok,lib.resolve_shadow t s u ok,?_⟩
  rw [(lib.resolve t).preserves s |>.streams]
  exact lib.resolve_ready t s u ok ready

theorem materialized_rebase (api : API α Store Byte) (neg : α → α)
    (e : Effect api neg caller Host Value) (initial current : State Store Host)
    (shadow : (Nat → ℝ) → (Nat → ℝ)) (original tensor : Tensor)
    (present : initial.live original)
    (h : Materialized api neg original (e.call.run initial).1 current shadow tensor) :
    Materialized api neg original initial current
      (fun m => shadow ((e.shadow initial).run m)) tensor := by
  refine ⟨h.shape,h.width,h.positive,h.present,?_,?_,h.ready⟩
  · intro index hi
    exact (h.values index hi).trans ((e.preserves initial).values original present index hi)
  · intro m index hi
    exact (h.realValues ((e.shadow initial).run m) index hi).trans
      ((e.shadow initial).values m original present index hi)

theorem transfer_materialized (api : API α Store Byte) (neg : α → α)
    (lib : APIHost api neg caller Host) (initial current : State Store Host)
    (shadow : (Nat → ℝ) → (Nat → ℝ)) (original tensor output : Tensor) (device : Nat)
    (h : Materialized api neg original initial current shadow tensor)
    (ok : ((lib.transfer device tensor).call.run current).2=.ok output) :
    Materialized api neg original initial ((lib.transfer device tensor).call.run current).1
      (fun m => ((lib.transfer device tensor).shadow current).run (shadow m)) output := by
  obtain ⟨hs,hw,hd,hn,hp⟩ := lib.transfer_metadata device tensor current output ok
  refine ⟨hs.trans h.shape,hw.trans h.width,hn.trans h.positive,hp,?_,?_,?_⟩
  · intro index hi
    have ht : Coordinate tensor index := by simpa only [Coordinate,h.shape] using hi
    exact (lib.transfer_values device tensor current output ok index ht).trans (h.values index hi)
  · intro m index hi
    have ht : Coordinate tensor index := by simpa only [Coordinate,h.shape] using hi
    exact (lib.transfer_shadow device tensor current output ok (shadow m) index ht).trans (h.realValues m index hi)
  · rw [(lib.transfer device tensor).preserves current |>.streams,hd]
    exact lib.transfer_ready device tensor current output ok h.ready

#axiomsClean resolve_start
#axiomsClean materialized_rebase
#axiomsClean transfer_materialized
#axiomsClean materialized_preserved
#guard_msgs (drop info) in
#auditModuleAxioms
end
end Torch
end VeriTile.Bench.Repairs.LayernormFwdTriton
