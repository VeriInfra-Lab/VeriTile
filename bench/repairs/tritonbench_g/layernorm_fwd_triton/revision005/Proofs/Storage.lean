import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.Torch

/-! Actual owner/storage byte metadata. A tensor view does not replace its
storage by its reachable words: prefix, padding and odd tails remain in the
caller union. The imported row proofs are reused only within one owner. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTritonRevision005.Storage
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section
open scoped Classical

structure Backing where
  owner : Nat
  base : Nat
  nbytes : Nat
  finite : base+nbytes ≤ 2^64

abbrev Key := Nat × Nat

def Backing.contains (s : Backing) (k : Key) : Prop :=
  k.1=s.owner ∧ s.base ≤ k.2 ∧ k.2 < s.base+s.nbytes

/-- `view.allocation.base` is the actual data pointer, while backing.base is
actual untyped_storage().data_ptr(). Their difference is retained explicitly. -/
structure Tensor where
  view : Torch.Tensor
  backing : Backing
  storageOffset : Nat
  owner : view.device=backing.owner
  fp32 : view.dtype=0
  pointer : (∃ index, Torch.Coordinate view index) →
    view.allocation.base=backing.base+storageOffset*4
  span : (∃ index, Torch.Coordinate view index) →
    storageOffset*4+view.allocation.extent*4 ≤ backing.nbytes
  emptyPointer : (¬∃ index, Torch.Coordinate view index) → view.allocation.base=0

abbrev World (Store : Type) := Nat → Store

def address (t : Tensor) (index : List Nat) : Nat := Torch.address t.view index

def contains (t : Tensor) (key : Key) : Prop := t.backing.contains key

def caller (x w : Tensor) (key : Key) : Prop := contains x key ∨ contains w key

def observe (api : Memory.API UInt32 Store UInt8) (m : World Store) (key : Key) : UInt8 :=
  api.byte (m key.1) key.2

def Frame (api : Memory.API UInt32 Store UInt8) (x w : Tensor)
    (before after : World Store) : Prop :=
  ∀ key, caller x w key → observe api after key=observe api before key

/-- Every live input word and all four of its bytes lie in the complete
storage. The backing is allowed to begin at every byte alignment and end with
one, two or three non-word bytes. -/
theorem coordinate_bytes (t : Tensor) (index : List Nat)
    (valid : Torch.Coordinate t.view index) (byte : Nat) (small : byte<4) :
    t.backing.contains (t.view.device,address t index+byte) := by
  have bound := t.view.bounds index valid
  have width := t.view.fp32Width t.fp32
  have span := t.span ⟨index,valid⟩
  have pointer := t.pointer ⟨index,valid⟩
  refine ⟨t.owner,?_,?_⟩ <;>
    simp only [address,Torch.address,VeriTile.Bench.Repairs.LayerNormFwd.Address.byteAddress,width,pointer]
  · omega
  · omega

/-- The frame directly includes storage bytes that no logical coordinate
names. This is a projection of the explicit storage union, not a free predicate. -/
theorem frame_x_storage (api : Memory.API UInt32 Store UInt8) (x w : Tensor)
    (before after : World Store) (frame : Frame api x w before after)
    (offset : Nat) (bound : offset<x.backing.nbytes) :
    observe api after (x.backing.owner,x.backing.base+offset)=
      observe api before (x.backing.owner,x.backing.base+offset) := by
  apply frame
  exact Or.inl ⟨rfl,by omega,by omega⟩

theorem frame_w_storage (api : Memory.API UInt32 Store UInt8) (x w : Tensor)
    (before after : World Store) (frame : Frame api x w before after)
    (offset : Nat) (bound : offset<w.backing.nbytes) :
    observe api after (w.backing.owner,w.backing.base+offset)=
      observe api before (w.backing.owner,w.backing.base+offset) := by
  apply frame
  exact Or.inr ⟨rfl,by omega,by omega⟩

/-- Four scalar byte loads and the source's little-endian integer packing.
This is representation, with no floating arithmetic or negation identity. -/
def pack (b0 b1 b2 b3 : UInt8) : UInt32 :=
  b0.toUInt32 ||| (b1.toUInt32 <<< 8) ||| (b2.toUInt32 <<< 16) ||| (b3.toUInt32 <<< 24)

def byteRead (api : Memory.API UInt32 Store UInt8) (m : Store) (a : Nat) : UInt32 :=
  pack (api.byte m a) (api.byte m (a+1)) (api.byte m (a+2)) (api.byte m (a+3))

/-- Ordinary local little-endian word/byte memory representation. It allows
unaligned mathematical observations; native source accesses these via bytes. -/
structure Codec (api : Memory.API UInt32 Store UInt8) : Prop where
  read_bytes : ∀ m a, byteRead api m a=api.read m a

#axiomsClean coordinate_bytes
#axiomsClean frame_x_storage
#axiomsClean frame_w_storage
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayernormFwdTritonRevision005.Storage
