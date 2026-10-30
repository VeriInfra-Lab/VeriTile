import bench.tritonbench_g.softmax_triton1.Proofs.Grid
import Mathlib.Data.BitVec

namespace VeriTile.Bench.Repairs.SoftmaxTriton1.Native

/-- Ordinary allocation metadata. Capacity is a byte count and may include
arbitrary prefix, suffix and holes beyond a logical view. -/
structure Buffer where
  base : Nat
  capacity : Nat
  device : Nat

def Buffer.Legal (b : Buffer) : Prop := b.base+b.capacity≤2^64 ∧ b.capacity<2^63

/-- A legal FP32 torch descriptor: sizes/strides/storage offset use nonnegative
signed int64 metadata; every active four-byte cell lies in its allocation.
There is no task-specific safe-size bound and no non-overlap demand on inputs. -/
def Descriptor (b : Buffer) (t : Tensor) : Prop :=
  t.device=b.device ∧ t.rows<2^63 ∧ t.cols<2^63 ∧
  t.offset<2^63 ∧ t.rowStride<2^63 ∧ t.colStride<2^63 ∧
  ∀ r c, r<t.rows → c<t.cols → 4*(t.index r c+1)≤b.capacity

/-- Actual widened source multiplies and adds. Pointer offset includes storage
base exactly once; data_ptr already starts at the descriptor's storage offset. -/
def offset (t : Tensor) (r c : Nat) : BitVec 64 :=
  BitVec.ofNat 64 t.offset + BitVec.ofNat 64 r * BitVec.ofNat 64 t.rowStride +
    BitVec.ofNat 64 c * BitVec.ofNat 64 t.colStride

def pointer (b : Buffer) (t : Tensor) (r c : Nat) : BitVec 64 :=
  BitVec.ofNat 64 b.base + offset t r c * BitVec.ofNat 64 4

theorem active_exact (b : Buffer) (t : Tensor) (legal : b.Legal)
    (descriptor : Descriptor b t) (r c : Nat) (hr : r<t.rows) (hc : c<t.cols) :
    (offset t r c).toNat=t.index r c ∧
    (pointer b t r c).toNat=b.base+4*t.index r c ∧
    (pointer b t r c).toNat+4≤b.base+b.capacity ∧
    r*t.rowStride<2^63 ∧ c*t.colStride<2^63 := by
  have cap := descriptor.2.2.2.2.2.2 r c hr hc
  have capBound := legal.2
  have low : t.index r c<2^63 := by omega
  have low64 : t.index r c<2^64 := by omega
  have ptrBound : b.base+4*t.index r c<2^64 := by
    have := legal.1
    omega
  have off : offset t r c=BitVec.ofNat 64 (t.index r c) := by
    simp [offset,Tensor.index,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  have ptr : (pointer b t r c).toNat=b.base+4*t.index r c := by
    rw [pointer,off,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
    rw [Nat.mul_comm (t.index r c) 4,Nat.mod_eq_of_lt ptrBound]
  refine ⟨?_,ptr,?_,?_,?_⟩
  · rw [off,BitVec.toNat_ofNat,Nat.mod_eq_of_lt low64]
  · rw [ptr];omega
  · dsimp [Tensor.index] at low;omega
  · dsimp [Tensor.index] at low;omega

/-- Local compiler/runtime ABI legality: program_id and arange are signed i32
before the source's explicit sign-extension. This is a per-local-call fact;
compatibility of an entire Python launch grid with a device remains deferred. -/
theorem i32_to_i64_exact (n : Nat) (legal : n<2^31) :
    ((BitVec.ofNat 32 n).signExtend 64).toInt=(n : Int) := by
  have hn : (BitVec.ofNat 32 n).toNat=n := Nat.mod_eq_of_lt (by omega)
  rw [BitVec.toInt_signExtend_of_le (by decide),
    BitVec.toInt_eq_toNat_of_lt (by rw [hn];omega),hn]


def index64 (n : Nat) : BitVec 64 := (BitVec.ofNat 32 n).signExtend 64

theorem index64_exact (n : Nat) (h : n<2^31) : index64 n=BitVec.ofNat 64 n := by
  apply BitVec.eq_of_toInt_eq
  have hn : (BitVec.ofNat 64 n).toNat=n := Nat.mod_eq_of_lt (by omega)
  rw [index64,i32_to_i64_exact n h,BitVec.toInt_eq_toNat_of_lt (by rw [hn];omega),hn]

def sourcePointer (b : Buffer) (t : Tensor) (r c : Nat) : BitVec 64 :=
  BitVec.ofNat 64 b.base +
    (BitVec.ofNat 64 t.offset + index64 r * BitVec.ofNat 64 t.rowStride +
      index64 c * BitVec.ofNat 64 t.colStride) * BitVec.ofNat 64 4

theorem source_pointer_exact (b : Buffer) (t : Tensor) (legal : b.Legal)
    (descriptor : Descriptor b t) (r c : Nat) (hr : r<t.rows) (hc : c<t.cols)
    (rABI : r<2^31) (cABI : c<2^31) :
    (sourcePointer b t r c).toNat=b.base+4*t.index r c ∧
    (sourcePointer b t r c).toNat+4≤b.base+b.capacity := by
  simp only [sourcePointer,index64_exact r rABI,index64_exact c cABI]
  exact ⟨(active_exact b t legal descriptor r c hr hc).2.1,
    (active_exact b t legal descriptor r c hr hc).2.2.1⟩

theorem int64_literal_exact (n : Nat) (h : n<2^63) : (BitVec.ofNat 64 n).toInt=(n : Int) := by
  have hn : (BitVec.ofNat 64 n).toNat=n := Nat.mod_eq_of_lt (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn];omega),hn]

theorem source_mask_exact (n c : Nat) (columnABI : c<2^31) (descriptor : n<2^63) :
    (index64 c).slt (BitVec.ofNat 64 n)=decide (c<n) := by
  simp [BitVec.slt,index64,i32_to_i64_exact c columnABI,int64_literal_exact n descriptor]

/-- The literal source pointer chain: torch data_ptr, row addptr, then column
addptr. This retains each fixed-width intermediate rather than presupposing an
unbounded flattened address. -/
def sequentialSourcePointer (b : Buffer) (t : Tensor) (r c : Nat) : BitVec 64 :=
  ((BitVec.ofNat 64 b.base + BitVec.ofNat 64 t.offset * BitVec.ofNat 64 4) +
    (index64 r * BitVec.ofNat 64 t.rowStride) * BitVec.ofNat 64 4) +
      (index64 c * BitVec.ofNat 64 t.colStride) * BitVec.ofNat 64 4

theorem source_sequence_same (b : Buffer) (t : Tensor) (r c : Nat) :
    sequentialSourcePointer b t r c=sourcePointer b t r c := by
  simp only [sequentialSourcePointer,sourcePointer,BitVec.add_mul,BitVec.add_assoc]

#axiomsClean active_exact
#axiomsClean i32_to_i64_exact
#axiomsClean source_pointer_exact
#axiomsClean source_mask_exact
#axiomsClean source_sequence_same
end VeriTile.Bench.Repairs.SoftmaxTriton1.Native
