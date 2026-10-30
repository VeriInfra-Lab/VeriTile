import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeControl

/-! Ordinary allocated typed cells. The signed64 allocation-byte bound belongs
to the allocator ABI; pointer identity/base and element size are retained. This
is a local address interpretation, not a byte codec or an execution oracle. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.Allocation
open VeriTile.Triton Address

def bytes : NativeSource.Storage → Nat
  | .fp32 => 4 | .fp16 | .bf16 => 2

structure Buffer where
  objectId : Nat
  allocationId : Nat
  base : Nat
  extent : Nat
  dtype : NativeSource.Storage
  representable : extent * bytes dtype ≤ 2^63

def Buffer.address (a : Buffer) (offset : Nat) : Nat := a.base + offset * bytes a.dtype

def Buffer.nativeAddress (a : Buffer) (offset : BitVec 64) : Nat :=
  a.base + (offset * word (bytes a.dtype)).toInt.toNat

theorem bytes_positive (d : NativeSource.Storage) : 0 < bytes d := by cases d <;> decide

theorem Buffer.extent_bound (a : Buffer) : a.extent ≤ 2^63 := by
  have hp := bytes_positive a.dtype
  have h := a.representable
  have hm : a.extent ≤ a.extent*bytes a.dtype := Nat.le_mul_of_pos_right _ hp
  omega

theorem Buffer.byte_exact (a : Buffer) (i : Nat) (hi : i<a.extent) :
    (word i * word (bytes a.dtype)).toInt = (i*bytes a.dtype : Nat) := by
  rw [word,word,←BitVec.ofNat_mul]
  exact word_exact (lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right hi (bytes_positive _)) a.representable)

theorem Buffer.native_address (a : Buffer) (i : Nat) (hi : i<a.extent) :
    a.nativeAddress (word i)=a.address i := by
  simp only [Buffer.nativeAddress,Buffer.address,a.byte_exact i hi,Int.toNat_natCast]

/-- Ordinary typed-cell/address contract, applicable to any initial or final
state. It relates just one buffer's cells to its native base and byte offsets. -/
structure Cells (α : Type) (a : Buffer) where
  typed : Nat → α
  physical : Nat → α
  coordinate : ∀ i, i<a.extent → physical (a.address i)=typed i

theorem native_cell {α : Type} {a : Buffer} (m : Cells α a) (i : Nat) (hi : i<a.extent) :
    m.physical (a.nativeAddress (word i))=m.typed i := by
  rw [a.native_address i hi]
  exact m.coordinate i hi

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGatedAttention.Allocation
