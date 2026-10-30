import bench.tritonbench_g.fused_rwkv6_kernel.Sources.Kernels
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel
noncomputable section
set_option maxHeartbeats 4000000

/-- Independent logical input arrays. No launch, program, or mutable output occurs. -/
structure Input (α : Type) where
  q : Nat → Nat → Nat → α
  k : Nat → Nat → Nat → α
  v : Nat → Nat → Nat → α
  w : Nat → Nat → Nat → α
  u : Nat → Nat → α
  h0 : Nat → Nat → Nat → α

def inputs {α : Type} (c : Shape) (m : Memory α) : Input α where
  q := fun z t e => m .q (inputKey c z t e)
  k := fun z t e => m .k (inputKey c z t e)
  v := fun z t f => m .v (inputValue c z t f)
  w := fun z t e => m .w (inputKey c z t e)
  u := fun z e => m .u (biasAddress c z e)
  h0 := fun z e f => m .h0 (stateAddress c z e f)

/-- Source-order padded scalar recurrence for bit-level/native interpretation.
The independent arrays are read directly; current memory is absent. -/
def scalarSeed {α : Type} (a : Ops α) (c : Shape) (x : Input α) (z e f : Nat) : α :=
  if c.initial then a.add a.zero (a.loadCast (if e<c.K ∧ f<c.V then x.h0 z e f else a.zero)) else a.zero

def scalarOuter {α : Type} (a : Ops α) (c : Shape) (x : Input α) (z n e f : Nat) :=
  a.mul (a.loadCast (if e<c.K then x.k z (visit c n) e else a.zero))
    (a.loadCast (if f<c.V then x.v z (visit c n) f else a.zero))

def scalarState {α : Type} (a : Ops α) (c : Shape) (x : Input α) (z e f : Nat) : Nat → α
  | 0 => scalarSeed a c x z e f
  | n+1 => a.add (a.mul (scalarState a c x z e f n)
      (a.exp (a.loadCast (if e<c.K then x.w z (visit c n) e else a.zero))))
      (scalarOuter a c x z n e f)

def scalarTerm {α : Type} (a : Ops α) (c : Shape) (x : Input α) (scale : α)
    (z n e f : Nat) :=
  a.mul (a.add (scalarState a c x z e f n)
    (a.mul (scalarOuter a c x z n e f)
      (a.loadCast (if e<c.K then x.u z e else a.zero))))
    (a.mul (a.loadCast (if e<c.K then x.q z (visit c n) e else a.zero)) scale)

def scalarPartial {α : Type} (a : Ops α) (c : Shape) (plan : Plans c) (x : Input α)
    (scale : α) (z i n f : Nat) :=
  (plan.key z i n f).eval a.zero a.add (fun j => scalarTerm a c x scale z n (i*c.BK+j) f)

def scalarOutput {α : Type} (a : Ops α) (c : Shape) (plan : Plans c) (x : Input α)
    (scale : α) (z t f : Nat) :=
  a.storeCast ((plan.host z t f).eval a.zero a.add (fun i =>
    a.storeCast (scalarPartial a c plan x scale z i (visit c t) f)))

/-- Independent result indexed by actual contiguous output addresses. -/
def target {α : Type} (a : Ops α) (c : Shape) (plan : Plans c) (x : Input α)
    (scale : α) (r : Region) (b : Nat) : α :=
  match r with
  | .o => a.storeCast (scalarPartial a c plan x scale (b/c.V/c.T%c.Z)
      (b/c.V/c.T/c.Z) (visit c (b/c.V%c.T)) (b%c.V))
  | .ht => a.storeCast (scalarState a c x (b/c.V/c.K) (b/c.V%c.K) (b%c.V) c.T)
  | .result => scalarOutput a c plan x scale (b/c.V/c.T) (b/c.V%c.T) (b%c.V)
  | _ => a.zero

/-- Input agreement is explicitly confined to read-only regions. -/
def InputRegion (r : Region) : Prop := r=.q ∨ r=.k ∨ r=.v ∨ r=.w ∨ r=.u ∨ r=.h0

def InputsEq {α : Type} (m n : Memory α) : Prop := ∀ r, InputRegion r → ∀ b, m r b=n r b

theorem inputs_eq {α : Type} (c : Shape) {m n : Memory α} (h : InputsEq m n) : inputs c m=inputs c n := by
  have hq := h .q (by simp [InputRegion])
  have hk := h .k (by simp [InputRegion])
  have hv := h .v (by simp [InputRegion])
  have hw := h .w (by simp [InputRegion])
  have hu := h .u (by simp [InputRegion])
  have hs := h .h0 (by simp [InputRegion])
  simp only [inputs,hq,hk,hv,hw,hu,hs]


end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel
