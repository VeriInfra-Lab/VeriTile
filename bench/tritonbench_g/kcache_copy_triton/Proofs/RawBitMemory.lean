import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

/-! A separate, width-tagged word-memory representation. This is not a byte
address/endian or hardware ABI model. The legacy `.real` channel is indexed by
an arbitrary type R; the fp32 source profile takes R = BitVec 32 explicitly.
Other floating cell variants have their own 8/16/32-bit payload widths.
No floating decode, rounding, finite-input premise or numerical callback occurs.
-/
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.RawBits
open VeriTile.Triton FP.Structural
noncomputable section
universe u

inductive RawCell (R : Type) where
  | real (value : R)
  | fp32 (bits : BitVec 32)
  | fp16 (bits : BitVec 16)
  | bf16 (bits : BitVec 16)
  | f8e4 (bits : BitVec 8)
  | f8e5 (bits : BitVec 8)
  | nat (value : Nat)
  | int (value : Int)
  | bool (value : Bool)
  | ptr (value : RegionName × Nat)
  | blockPtr (value : BlockPtr)

/-- A logical carrier. The word constructor packs exact finite-width words;
the legacy constructor preserves arbitrary legacy-channel values. Neither side
is an IEEE numerical projection. The common zero permits total wrong-tag
reads to remain canonical, without assuming that input cells have one tag. -/
inductive Carrier (R : Type) where
  | legacy (value : R)
  | word (bits : BitVec 32)
instance {R : Type} : Inhabited (Carrier R) := ⟨Carrier.word 0⟩

private def encodeReal {R : Type} [Inhabited R] (r : R) : Carrier R := by
  classical
  exact if r=default then Carrier.word 0 else Carrier.legacy r
private def decodeReal {R : Type} [Inhabited R] : Carrier R → R
  | .legacy r => r
  | .word _ => default
private def word32 {R : Type} : Carrier R → BitVec 32
  | .word b => b
  | .legacy _ => 0

def encode {R : Type} [Inhabited R] : RawCell R → Cell (Carrier R)
  | .real r => .mk .real (encodeReal r)
  | .fp32 b => .mk .fp32 (.word b)
  | .fp16 b => .mk .fp16 (.word (b.zeroExtend 32))
  | .bf16 b => .mk .bf16 (.word (b.zeroExtend 32))
  | .f8e4 b => .mk .f8e4 (.word (b.zeroExtend 32))
  | .f8e5 b => .mk .f8e5 (.word (b.zeroExtend 32))
  | .nat n => .mk .nat n
  | .int n => .mk .int n
  | .bool b => .mk .bool b
  | .ptr p => .mk .ptr p
  | .blockPtr p => .mk .blockPtr p

def decode {R : Type} [Inhabited R] : Cell (Carrier R) → RawCell R
  | .mk .real r => .real (decodeReal r)
  | .mk .fp32 b => .fp32 (word32 b)
  | .mk .fp16 b => .fp16 ((word32 b).truncate 16)
  | .mk .bf16 b => .bf16 ((word32 b).truncate 16)
  | .mk .f8e4 b => .f8e4 ((word32 b).truncate 8)
  | .mk .f8e5 b => .f8e5 ((word32 b).truncate 8)
  | .mk .nat n => .nat n
  | .mk .int n => .int n
  | .mk .bool b => .bool b
  | .mk .ptr p => .ptr p
  | .mk .blockPtr p => .blockPtr p

@[simp] theorem decode_encode {R : Type} [Inhabited R] (c : RawCell R) : decode (encode c) = c := by
  cases c <;> simp [encode,decode,word32]
  case real value =>
    by_cases h : value=default <;> simp [encodeReal,decodeReal,h]

theorem encode_injective {R : Type} [Inhabited R] : Function.Injective (@encode R _) := by
  intro a b h
  simpa using congrArg decode h

/-- The reverse round-trip is stated on the exact image. Arbitrary ill-tagged
Structural carriers are not falsely equated with finite-width raw cells. -/
def Canonical {R : Type} [Inhabited R] (c : Cell (Carrier R)) : Prop := ∃ r, encode r = c

theorem encode_decode_of_canonical {R : Type} [Inhabited R] (c : Cell (Carrier R))
    (h : Canonical c) : encode (decode c) = c := by
  obtain ⟨r,rfl⟩ := h
  simp

abbrev Memory (R : Type) := RegionName → Nat → RawCell R

def encodeMemory {R : Type} [Inhabited R] (m : Memory R) : RegionName → Nat → Cell (Carrier R) :=
  fun r o => encode (m r o)
def decodeMemory {R : Type} [Inhabited R] (m : RegionName → Nat → Cell (Carrier R)) : Memory R :=
  fun r o => decode (m r o)

@[simp] theorem decode_encode_memory {R : Type} [Inhabited R] (m : Memory R) :
    decodeMemory (encodeMemory m) = m := by
  funext r o
  simp [decodeMemory,encodeMemory]

def write {R : Type} (m : Memory R) (r : RegionName) (o : Nat) (c : RawCell R) : Memory R :=
  fun r' o' => if r'=r ∧ o'=o then c else m r' o'

theorem encode_write {R : Type} [Inhabited R] (m : Memory R) (r : RegionName) (o : Nat) (c : RawCell R) :
    encodeMemory (write m r o c) =
      fun r' o' => if r'=r ∧ o'=o then encode c else encodeMemory m r' o' := by
  funext r' o'
  simp only [encodeMemory,write]
  split <;> rfl

theorem decode_write {R : Type} [Inhabited R] (s : State (Carrier R))
    (r : RegionName) (o : Nat) (c : Cell (Carrier R)) :
    decodeMemory (s.write r o c).mem = write (decodeMemory s.mem) r o (decode c) := by
  funext r' o'
  simp only [decodeMemory,State.write,write]
  split <;> rfl

/-- Source payload extraction, including the old default-on-wrong-tag rule.
No well-typed-input hypothesis or homogeneous-memory condition is added. -/
def readStorage {R : Type} [Inhabited R] : TypedCopy.Storage → RawCell R → RawCell R
  | .fp16, .fp16 b => .fp16 b
  | .fp16, _ => .fp16 0
  | .bf16, .bf16 b => .bf16 b
  | .bf16, _ => .bf16 0
  | .f8e4, .f8e4 b => .f8e4 b
  | .f8e4, _ => .f8e4 0
  | .f8e5, .f8e5 b => .f8e5 b
  | .f8e5, _ => .f8e5 0
  | .fp32, .real r => .real r
  | .fp32, _ => .real default
  | .fp64, .real r => .real r
  | .fp64, _ => .real default

@[simp] theorem read_storage_encode {R : Type} [Inhabited R]
    (d : TypedCopy.Storage) (c : RawCell R) :
    Cell.mk d.tag ((encode c).read d.tag) = encode (readStorage d c) := by
  cases d <;> cases c <;>
    simp [TypedCopy.Storage.tag,encode,readStorage,Cell.read,defaultValue,encodeReal] <;> rfl

/-- Nat metadata keeps exactly the old unbounded Nat/default semantics. -/
def readNat {R : Type} : RawCell R → Nat
  | .nat n => n
  | _ => 0

@[simp] theorem read_nat_encode {R : Type} [Inhabited R] (c : RawCell R) :
    (encode c).read .nat = readNat c := by
  cases c <;> simp [encode,Cell.read,readNat,defaultValue]

/-- A source fp32 load/store uses the legacy `.real` carrier slot. It does not
use the distinct Structural `.fp32` cell variant. This mapping is explicit. -/
theorem fp32_source_storage_mapping (b : BitVec 32) :
    readStorage .fp32 (RawCell.real b) = RawCell.real b ∧
    TypedCopy.Storage.fp32.tag = TileDType.real := ⟨rfl,rfl⟩

#axiomsClean decode_encode
#axiomsClean encode_injective
#axiomsClean encode_decode_of_canonical
#axiomsClean decode_encode_memory
#axiomsClean encode_write
#axiomsClean decode_write
#axiomsClean read_storage_encode
#axiomsClean read_nat_encode
#axiomsClean fp32_source_storage_mapping
end
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.RawBits
