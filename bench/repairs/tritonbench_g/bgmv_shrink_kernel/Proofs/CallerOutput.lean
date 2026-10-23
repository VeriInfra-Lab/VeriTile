import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Snapshot
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.ViewGuard

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.CallerOutput
open VeriTile.Triton
namespace Shared
abbrev View := VeriTile.Bench.Repairs.BgmvExpandSlice.ReturnedStorage.View
abbrev Coord := VeriTile.Bench.Repairs.BgmvExpandSlice.ReturnedStorage.Coord
abbrev Bytes := VeriTile.Bench.Repairs.BgmvExpandSlice.ReturnedStorage.Bytes
abbrev Admitted := VeriTile.Bench.Repairs.BgmvExpandSlice.ViewGuard.Admitted
abbrev writable := VeriTile.Bench.Repairs.BgmvExpandSlice.ViewGuard.writable
end Shared
noncomputable section
open Classical

/-- Original supplied output descriptor plus its actual scalar storage width.
The reused exact gcd guard proves logical injectivity for fp16/bf16/fp32 alike. -/
structure View where
  original : Shared.View
  width : Nat
  positive : 0<width
  backed : ∀ i : Shared.Coord original,width*original.offset i+width≤original.allocationBytes

abbrev Coord (v : View) := Shared.Coord v.original
abbrev Bytes := Shared.Bytes

def address (v : View) (i : Coord v) (byte : Fin v.width) :=
  v.original.allocationBase+v.original.offset i*v.width+byte.val

def Active (h : Call) (v : View) (indices : Nat → Int) (i : Coord v) : Prop :=
  i.1.val<h.batches ∧ i.2.val<h.N ∧ indices i.1.val≠-1

theorem byte_injective (v : View) (guard : Shared.Admitted v.original)
    (i j : Coord v) (a b : Fin v.width) (same : address v i a=address v j b) : i=j ∧ a=b := by
  have values : v.original.offset i*v.width+a.val=v.original.offset j*v.width+b.val := by
    unfold address at same
    omega
  have words := congrArg (fun x => x/v.width) values
  dsimp only at words
  rw [Nat.add_comm (v.original.offset i*v.width),Nat.add_comm (v.original.offset j*v.width),
    Nat.add_mul_div_right _ _ v.positive,Nat.add_mul_div_right _ _ v.positive,
    Nat.div_eq_of_lt a.isLt,Nat.div_eq_of_lt b.isLt] at words
  simp only [Nat.zero_add] at words
  exact ⟨Shared.writable _ guard words,Fin.ext (by rw [words] at values; omega)⟩

def copy {α : Type} (h : Call) (v : View) (encode : Bool → α → Fin v.width → BitVec 8)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Bytes) : Bytes :=
  fun owner ptr => if found : ∃ i : Coord v,∃ byte : Fin v.width,
      Active h v indices i ∧ owner=v.original.owner ∧ ptr=address v i byte then
    let i := Classical.choose found
    let byte := Classical.choose (Classical.choose_spec found)
    encode v.original.negative (scratch i.1.val i.2.val) byte
  else before owner ptr

theorem copy_byte {α : Type} (h : Call) (v : View)
    (encode : Bool → α → Fin v.width → BitVec 8) (guard : Shared.Admitted v.original)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Bytes)
    (i : Coord v) (active : Active h v indices i) (byte : Fin v.width) :
    copy h v encode indices scratch before v.original.owner (address v i byte)=
      encode v.original.negative (scratch i.1.val i.2.val) byte := by
  let found : ∃ j : Coord v,∃ b : Fin v.width,
      Active h v indices j ∧ v.original.owner=v.original.owner ∧ address v i byte=address v j b := ⟨i,byte,active,rfl,rfl⟩
  have same := (Classical.choose_spec (Classical.choose_spec found)).2.2
  obtain ⟨coord,byteEq⟩ := byte_injective v guard i (Classical.choose found) byte
    (Classical.choose (Classical.choose_spec found)) same
  unfold copy
  rw [dif_pos found]
  change encode v.original.negative
    (scratch (Classical.choose found).1.val (Classical.choose found).2.val)
    (Classical.choose (Classical.choose_spec found))=_
  rw [←byteEq,←coord]

theorem frame {α : Type} (h : Call) (v : View) (encode : Bool → α → Fin v.width → BitVec 8)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Bytes)
    (owner ptr : Nat) (outside : ¬∃ i : Coord v,∃ byte : Fin v.width,
      Active h v indices i ∧ owner=v.original.owner ∧ ptr=address v i byte) :
    copy h v encode indices scratch before owner ptr=before owner ptr := by
  simp only [copy,dif_neg outside]

theorem inactive {α : Type} (h : Call) (v : View) (encode : Bool → α → Fin v.width → BitVec 8)
    (guard : Shared.Admitted v.original) (indices : Nat → Int) (scratch : Nat → Nat → α)
    (before : Bytes) (i : Coord v) (no : ¬Active h v indices i) (byte : Fin v.width) :
    copy h v encode indices scratch before v.original.owner (address v i byte)=
      before v.original.owner (address v i byte) := by
  apply frame
  rintro ⟨j,b,active,_,same⟩
  have equal := (byte_injective v guard i j byte b same).1
  exact no (equal.symm ▸ active)

structure Returned where
  result : Unit
  view : View
  memory : Bytes

def returned {α : Type} (h : Call) (v : View) (encode : Bool → α → Fin v.width → BitVec 8)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Bytes) : Returned :=
  ⟨(),v,copy h v encode indices scratch before⟩

theorem identity {α : Type} (h : Call) (v : View) (encode : Bool → α → Fin v.width → BitVec 8)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Bytes) :
    (returned h v encode indices scratch before).result=() ∧
    (returned h v encode indices scratch before).view=v := ⟨rfl,rfl⟩

/-- Ordinary local destination confinement on a possibly partial final copy
failure. No all-or-nothing failure guarantee is asserted. -/
structure CopyFailure (ε : Type) (h : Call) (v : View) (indices : Nat → Int) (before : Bytes) where
  error : ε
  after : Bytes
  frame : ∀ owner ptr,(¬∃ i : Coord v,∃ byte : Fin v.width,
    Active h v indices i ∧ owner=v.original.owner ∧ ptr=address v i byte) →
    after owner ptr=before owner ptr

#axiomsClean byte_injective
#axiomsClean copy_byte
#axiomsClean frame
#axiomsClean inactive
#axiomsClean identity
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.CallerOutput
