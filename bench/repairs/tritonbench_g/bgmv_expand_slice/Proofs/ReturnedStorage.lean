import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Geometry
import Mathlib.Data.BitVec

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.ReturnedStorage
open VeriTile.Triton
noncomputable section
open Classical

/-- The original supplied output descriptor. Lazy negative views alter the
encoded value, while these strides and offsets select its actual backing. -/
structure View where
  object : Nat
  owner : Nat
  allocationBase : Nat
  allocationBytes : Nat
  storageOffset : Nat
  rows : Nat
  columns : Nat
  rowStride : Nat
  columnStride : Nat
  negative : Bool
  physicalWindow : allocationBase+allocationBytes≤2^64

abbrev Coord (v : View) := Fin v.rows × Fin v.columns

def View.offset (v : View) (i : Coord v) : Nat :=
  v.storageOffset+i.1.val*v.rowStride+i.2.val*v.columnStride

def View.address (v : View) (i : Coord v) (byte : Fin 4) : Nat :=
  v.allocationBase+4*v.offset i+byte.val

/-- Writeability is a property of the caller's output view, not an allocator
contract. The original contiguous domain has it; the repaired gcd guard also
allows nonoverlapping transposes, gaps and storage offsets. -/
def View.Writable (v : View) : Prop := Function.Injective v.offset

def View.Backed (v : View) : Prop := ∀ i : Coord v,4*v.offset i+4≤v.allocationBytes

theorem byte_injective (v : View) (writable : v.Writable) (i j : Coord v) (a b : Fin 4)
    (same : v.address i a=v.address j b) : i=j ∧ a=b := by
  unfold View.address at same
  have ha := a.isLt
  have hb := b.isLt
  have word : v.offset i=v.offset j := by omega
  exact ⟨writable word,Fin.ext (by omega)⟩

theorem backed_byte (v : View) (backed : v.Backed) (i : Coord v) (byte : Fin 4) :
    v.allocationBase≤v.address i byte ∧
    v.address i byte<v.allocationBase+v.allocationBytes ∧ v.address i byte<2^64 := by
  have capacity := backed i
  have lane := byte.isLt
  have window := v.physicalWindow
  unfold View.address
  omega

abbrev Bytes := Nat → Nat → BitVec 8

def Active (h : Call) (v : View) (indices : Nat → Int) (i : Coord v) : Prop :=
  i.1.val<h.batches ∧ h.slice≤i.2.val ∧ i.2.val<h.slice+h.N ∧ indices i.1.val≠-1

/-- Ordinary FP32 scalar store encoding, including the supplied view's lazy
negative flag. It is a local word/byte operation, never an output oracle. -/
abbrev Encode (α : Type) := Bool → α → Fin 4 → BitVec 8

/-- The index_select/index_copy pair copies active scratch coordinates into
exactly the same supplied output object. Values are the actual scratch reads.
This is the local coordinate-preserving tensor-library copy contract. -/
def copy {α : Type} (encode : Encode α) (h : Call) (v : View) (indices : Nat → Int)
    (scratch : Nat → Nat → α) (before : Bytes) : Bytes :=
  fun owner address => if found : ∃ i : Coord v, ∃ byte : Fin 4,
      Active h v indices i ∧ owner=v.owner ∧ address=v.address i byte then
    let i := Classical.choose found
    let byte := Classical.choose (Classical.choose_spec found)
    encode v.negative (scratch i.1.val (i.2.val-h.slice)) byte
  else before owner address

theorem copy_byte {α : Type} (encode : Encode α) (h : Call) (v : View)
    (writable : v.Writable) (indices : Nat → Int) (scratch : Nat → Nat → α)
    (before : Bytes) (i : Coord v) (active : Active h v indices i) (byte : Fin 4) :
    copy encode h v indices scratch before v.owner (v.address i byte)=
      encode v.negative (scratch i.1.val (i.2.val-h.slice)) byte := by
  let found : ∃ j : Coord v, ∃ b : Fin 4,
      Active h v indices j ∧ v.owner=v.owner ∧ v.address i byte=v.address j b := ⟨i,byte,active,rfl,rfl⟩
  have same := (Classical.choose_spec (Classical.choose_spec found)).2.2
  obtain ⟨coord,byteEq⟩ := byte_injective v writable i (Classical.choose found) byte
    (Classical.choose (Classical.choose_spec found)) same
  unfold copy
  rw [dif_pos found]
  change encode v.negative
    (scratch (Classical.choose found).1.val ((Classical.choose found).2.val-h.slice))
    (Classical.choose (Classical.choose_spec found))=_
  rw [←byteEq,←coord]

theorem copy_frame {α : Type} (encode : Encode α) (h : Call) (v : View)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Bytes)
    (owner address : Nat) (outside : ¬∃ i : Coord v, ∃ byte : Fin 4,
      Active h v indices i ∧ owner=v.owner ∧ address=v.address i byte) :
    copy encode h v indices scratch before owner address=before owner address := by
  simp only [copy,dif_neg outside]

theorem inactive_byte {α : Type} (encode : Encode α) (h : Call) (v : View)
    (writable : v.Writable) (indices : Nat → Int) (scratch : Nat → Nat → α)
    (before : Bytes) (i : Coord v) (inactive : ¬Active h v indices i) (byte : Fin 4) :
    copy encode h v indices scratch before v.owner (v.address i byte)=before v.owner (v.address i byte) := by
  apply copy_frame
  rintro ⟨j,b,active,_,same⟩
  have equal := (byte_injective v writable i j byte b same).1
  exact inactive (equal.symm ▸ active)


theorem empty {α : Type} (encode : Encode α) (h : Call) (v : View)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Bytes)
    (zero : h.batches=0 ∨ h.N=0) : copy encode h v indices scratch before=before := by
  funext owner address
  apply copy_frame
  rintro ⟨i,byte,active,_,_⟩
  rcases active with ⟨batch,lower,upper,_⟩
  rcases zero with zero | zero <;> omega

structure Returned where
  result : Unit
  outputObject : Nat
  outputOwner : Nat
  memory : Bytes

def returned {α : Type} (encode : Encode α) (h : Call) (v : View) (indices : Nat → Int)
    (scratch : Nat → Nat → α) (before : Bytes) : Returned :=
  ⟨(),v.object,v.owner,copy encode h v indices scratch before⟩

theorem identity {α : Type} (encode : Encode α) (h : Call) (v : View) (indices : Nat → Int)
    (scratch : Nat → Nat → α) (before : Bytes) :
    (returned encode h v indices scratch before).result=() ∧
    (returned encode h v indices scratch before).outputObject=v.object ∧
    (returned encode h v indices scratch before).outputOwner=v.owner := ⟨rfl,rfl,rfl⟩

#axiomsClean byte_injective
#axiomsClean copy_byte
#axiomsClean copy_frame
#axiomsClean identity
#axiomsClean inactive_byte
#axiomsClean empty
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.ReturnedStorage
