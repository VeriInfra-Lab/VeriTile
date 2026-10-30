import bench.tritonbench_g.vector_addition.Proofs.Mathematical_IntegerValueCorrect

/-! Explicit finite input/output words for the 81 integer/bool profiles.
Bool storage is raw i8; normalization happens only at the load boundary.
The result retains the complete output byte, including noncanonical bool
bytes. This module does not model native byte assembly or pointer validity. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.IntegerWords
open VeriTile.Triton FP.Structural
open IntegerSource (Canonical Storage castValue promote)
noncomputable section

def width (d : Canonical) : Nat := d.storage.storeFormat.width
abbrev Word (d : Canonical) := BitVec (width d)
abbrev Memory (α : Type) := RegionName → Nat → Cell α

/-- Complete input/output storage cells; bool uses signed i8 storage. -/
def decodeCell {α : Type} (d : Canonical) (v : Word d) : Cell α :=
  match d with
  | .b1 | .i8 | .i16 | .i32 | .i64 => .mk .int v.toInt
  | .u8 | .u16 | .u32 | .u64 => .mk .nat v.toNat

def loadedWord (d : Canonical) (v : Word d) : Int :=
  match d with
  | .b1 => if v = 0 then 0 else 1
  | .i8 | .i16 | .i32 | .i64 => v.toInt
  | .u8 | .u16 | .u32 | .u64 => (v.toNat : Int)

theorem word_toInt_zero_iff {n : Nat} (v : BitVec n) : v.toInt = 0 ↔ v = 0 := by
  constructor
  · intro h
    have same := congrArg (BitVec.ofInt n) h
    simpa using same
  · intro h
    subst v
    simp

theorem loaded_decode {α : Type} [Inhabited α] (d : Canonical) (v : Word d) :
    IntegerSource.loaded d.storage (decodeCell (α := α) d v) = loadedWord d v := by
  cases d <;> simp [IntegerSource.loaded, Canonical.storage, decodeCell,
    loadedWord, Cell.read, word_toInt_zero_iff]

theorem bool_load_zero (v : Word .b1) : loadedWord .b1 v = 0 ↔ v = 0 := by
  simp [loadedWord]

theorem bool_load_nonzero (v : Word .b1) : loadedWord .b1 v = 1 ↔ v ≠ 0 := by
  simp [loadedWord]

/-- Storage re-encoding reads the complete tagged cell; bool is not normalized. -/
def encodeCell {α : Type} [Inhabited α] (d : Canonical) (cell : Cell α) : Word d :=
  match d with
  | .b1 | .i8 | .i16 | .i32 | .i64 => BitVec.ofInt _ (cell.read .int)
  | .u8 | .u16 | .u32 | .u64 => BitVec.ofNat _ (cell.read .nat)

theorem storage_roundtrip {α : Type} [Inhabited α] (d : Canonical) (v : Word d) :
    encodeCell d (decodeCell (α := α) d v) = v := by
  cases d <;> simp [encodeCell, decodeCell, Cell.read]

theorem stored_decode {α : Type} (d : Canonical) (v : Int) :
    IntegerSource.stored (α := α) d.storage v = decodeCell d (BitVec.ofInt (width d) v) := by
  cases d <;> rfl

/-- Initial-word arithmetic, including promotion-width modular addition and
the exact output storage width. No cell read, memory or execution occurs. -/
def resultWord (x y : Canonical) (a : Word x) (b : Word y) : Word x :=
  let c := promote x.storage y.storage
  BitVec.ofInt (width x)
    (castValue c (castValue c (loadedWord x a) + castValue c (loadedWord y b)))

theorem result_refines {α : Type} [Inhabited α] (x y : Canonical)
    (a : Word x) (b : Word y) (p : ValueProofs.Parameters)
    (mem : Memory α) (o : Nat)
    (hx : mem p.x o = decodeCell x a) (hy : mem p.y o = decodeCell y b) :
    IntegerSource.resultCell x.storage y.storage p mem o = decodeCell x (resultWord x y a b) := by
  simp only [IntegerSource.resultCell, hx, hy, loaded_decode, resultWord, stored_decode]

/-- Noncanonical bool output remains a byte; it is normalized only if loaded
again as bool. This finite-word theorem distinguishes storage from loading. -/
theorem bool_integer_output_three :
    resultWord .b1 .i8 255 2 = 3 := by decide +kernel

theorem bool_both_nonzero_wrap (a b : Word .b1) (ha : a ≠ 0) (hb : b ≠ 0) :
    resultWord .b1 .b1 a b = 0 := by
  have hx := (bool_load_nonzero a).2 ha
  have hy := (bool_load_nonzero b).2 hb
  simp only [resultWord, hx, hy]
  decide +kernel

#specNonCircular resultWord avoiding [IntegerSource.source, IntegerSource.runGrid,
  IntegerSource.runGridPrefix, FP.Structural.exec, IntegerSource.resultCell]
#axiomsClean word_toInt_zero_iff
#axiomsClean loaded_decode
#axiomsClean bool_load_zero
#axiomsClean bool_load_nonzero
#axiomsClean storage_roundtrip
#axiomsClean stored_decode
#axiomsClean result_refines
#axiomsClean bool_integer_output_three
#axiomsClean bool_both_nonzero_wrap
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.IntegerWords
