import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.Source

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Storage
open VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd RawBoundary

/-- Ordinary allocator metadata, in bytes. Partial trailing bytes are retained. -/
structure Allocation where
  address : Nat
  bytes : Nat
  device : Nat
  endBound : address + bytes ≤ 2^64
  aligned : address % 4 = 0

/-- Actual same-call tensor object, allocation and pointer. Pointer identity is
required only when an access exists, covering empty Torch views with null ptr. -/
structure Tensor (alloc : Nat → Allocation) where
  object : Nat
  shape : List Nat
  stride : Nat → Nat
  dtype : VeriTile.Triton.FloatDType
  fp32 : dtype = .fp32
  view : View
  dataPtr : Nat
  device : Nat
  sameDevice : device = (alloc view.allocation).device
  pointer : ∀ off, 4*(view.base+off)+4 ≤ (alloc view.allocation).bytes →
    dataPtr = (alloc view.allocation).address+4*view.base

def Tensor.Covers (t : Tensor alloc) (off : Nat) : Prop :=
  4*(t.view.base+off)+4 ≤ (alloc t.view.allocation).bytes

def bytes (memory : PhysicalMemory Word) (allocation byte : Nat) : BitVec 8 :=
  BitVec.ofNat 8 ((memory allocation (byte/4)).toNat / 2^(8*(byte%4)))

def byte (w : Word) (i : Nat) : BitVec 8 := BitVec.ofNat 8 (w.toNat / 2^(8*i))

/-- All four bytes of a returned word use the same actual allocation/offset. -/
theorem read_bytes (memory : PhysicalMemory Word) (t : Tensor alloc) (off i : Nat) (hi : i < 4) :
    bytes memory t.view.allocation (4*(t.view.base+off)+i) =
      byte (t.view.read memory off) i := by
  simp [bytes,byte,View.read,Nat.add_div,Nat.add_mod,Nat.div_eq_of_lt hi,Nat.mod_eq_of_lt hi,Nat.not_le.mpr hi]

/-- No whole-word-validity premise: the final 1, 2 or 3 existing allocation
bytes are preserved through the virtual last word just like every other byte. -/
theorem all_byte_frame (before after : PhysicalMemory Word) (old : Nat → Prop)
    (frame : ∀ a off, old a → after a off = before a off)
    (a q : Nat) (ha : old a) : bytes after a q = bytes before a q := by
  simp only [bytes,frame a (q/4) ha]

theorem span (t : Tensor alloc) (off : Nat) (h : t.Covers off) :
    t.dataPtr+4*off+4 ≤ (alloc t.view.allocation).address+(alloc t.view.allocation).bytes ∧
    t.dataPtr+4*off+4 ≤ 2^64 := by
  have hp := t.pointer off h
  have he := (alloc t.view.allocation).endBound
  unfold Tensor.Covers at h
  omega

theorem native_address (t : Tensor alloc) (off : Nat) (h : t.Covers off) :
    (Index.address t.dataPtr (Index.u off)).toNat = t.dataPtr+4*off := by
  have hs := (span t off h).2
  change ((BitVec.ofNat 64 t.dataPtr) + BitVec.ofNat 64 off * BitVec.ofNat 64 4).toNat = _
  rw [← BitVec.ofNat_mul,← BitVec.ofNat_add]
  change (t.dataPtr+off*4) % 2^64 = _
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Storage
