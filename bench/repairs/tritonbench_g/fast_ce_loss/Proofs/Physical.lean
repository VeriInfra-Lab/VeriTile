import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.API

namespace VeriTile.Bench.Repairs.FastCeLoss.Physical
open Source Complete API
noncomputable section
set_option maxHeartbeats 3000000

abbrev Byte := BitVec 8

/-- Source cells use four-byte coordinates. Marker cells cover partial trailing
words too; numeric observations separately require all four bytes to exist. -/
structure Layout (σ : Type) where
  capacity : σ → Nat → Nat
  observe : σ → Nat → Nat → Byte

inductive Marker where
  | old (cell : Address)
  | changed

def initialMarkers : Memory Marker := Marker.old

def markerArithmetic : Arithmetic Marker where
  literal := fun _ => .changed
  natural := fun _ => .changed
  negInf := .changed
  load := id
  cast := id
  add := fun _ _ => .changed
  sub := fun _ _ => .changed
  mul := fun _ _ => .changed
  div := fun _ _ => .changed
  exp := fun _ => .changed
  log := fun _ => .changed
  tanh := fun _ => .changed
  maximum := fun _ => .changed
  sum := fun _ => .changed
  hostSum := fun _ => .changed
  hostLSE := fun _ => .changed
  hostAdd := fun _ _ => .changed
  hostDiv := fun _ _ => .changed
  upstreamDiv := fun _ _ => .changed

/-- Ordinary opaque-token representation. It relates every preserved source
cell to its original bytes, without a logits/caller/write-exclusion predicate. -/
def Representation {σ : Type} (layout : Layout σ) (initial final : σ) (source : Memory Marker) : Prop :=
  ∀ target origin, source target=.old origin → ∀ lane<4,
    4*origin.2+lane<layout.capacity initial origin.1 →
    4*target.2+lane<layout.capacity final target.1 ∧
    layout.observe final target.1 (4*target.2+lane)=layout.observe initial origin.1 (4*origin.2+lane)

def outside (s : Shape) (b : Buffers) (storage byte : Nat) : Prop :=
  storage≠b.logits.storage ∨ byte<4*b.logits.offset ∨ 4*(b.logits.offset+s.rows*s.vocab)≤byte

 theorem byte_frame {σ : Type} (physical : Layout σ) (initial final : σ)
    (o : Options) (s : Shape) (b : Buffers) (labels : Nat → Int) (items : Option Marker)
    (layout : ForwardLayout s b) (fresh : PrivateBuffers b)
    (represented : Representation physical initial final
      (API.backward markerArithmetic o s b labels items .changed initialMarkers).memory)
    (storage byte : Nat) (bound : byte<physical.capacity initial storage)
    (chunks : storage≠b.chunks.storage) (losses : storage≠b.losses.storage)
    (lse : storage≠b.lse.storage) (scalar : storage≠b.scalar.storage) (up : storage≠b.upstream.storage)
    (frame : outside s b storage byte) :
    byte<physical.capacity final storage ∧ physical.observe final storage byte=physical.observe initial storage byte := by
  have lane : byte%4<4 := Nat.mod_lt _ (by decide)
  have coordinate : 4*(byte/4)+byte%4=byte := by omega
  have sourceOutside : storage≠b.logits.storage ∨ byte/4<b.logits.offset ∨
      b.logits.offset+s.rows*s.vocab≤byte/4 := by
    rcases frame with frame|frame|frame
    · exact Or.inl frame
    · exact Or.inr (Or.inl (by omega))
    · exact Or.inr (Or.inr (by omega))
  have preserved := API.caller_frame markerArithmetic o s b labels items .changed initialMarkers
    layout fresh (storage,byte/4) chunks losses lse scalar up sourceOutside
  have h := represented (storage,byte/4) (storage,byte/4) preserved (byte%4) lane
    (by simpa only [coordinate] using bound)
  simpa only [coordinate] using h

#axiomsClean byte_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Physical
