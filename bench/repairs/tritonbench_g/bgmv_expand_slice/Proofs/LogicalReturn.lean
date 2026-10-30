import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.ReturnedStorage

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.LogicalReturn
open ReturnedStorage
noncomputable section
open Classical

/-- Ordinary logical read/write conversion for supplied negative views. Real
values are a mathematical memory interpretation, never finite FP32 bytes. -/
structure Codec (α : Type) where
  encode : Bool → α → α
  decode : Bool → α → α
  roundtrip : ∀ flag value,decode flag (encode flag value)=value

abbrev Words (α : Type) := Nat → Nat → α

def read {α : Type} (codec : Codec α) (view : View) (memory : Words α) (i : Coord view) : α :=
  codec.decode view.negative (memory view.owner (view.address i ⟨0,by decide⟩))

def copy {α : Type} (codec : Codec α) (h : Call) (view : View) (indices : Nat → Int)
    (scratch : Nat → Nat → α) (before : Words α) : Words α :=
  fun owner address => if found : ∃ i : Coord view,
      Active h view indices i ∧ owner=view.owner ∧ address=view.address i ⟨0,by decide⟩ then
    let i := Classical.choose found
    codec.encode view.negative (scratch i.1.val (i.2.val-h.slice))
  else before owner address

theorem readback {α : Type} (codec : Codec α) (h : Call) (view : View)
    (writable : view.Writable) (indices : Nat → Int) (scratch : Nat → Nat → α)
    (before : Words α) (i : Coord view) (active : Active h view indices i) :
    read codec view (copy codec h view indices scratch before) i=scratch i.1.val (i.2.val-h.slice) := by
  let found : ∃ j : Coord view,Active h view indices j ∧ view.owner=view.owner ∧
      view.address i ⟨0,by decide⟩=view.address j ⟨0,by decide⟩ := ⟨i,active,rfl,rfl⟩
  have same := (byte_injective view writable i (Classical.choose found) ⟨0,by decide⟩ ⟨0,by decide⟩
    (Classical.choose_spec found).2.2).1
  unfold read copy
  rw [dif_pos found,codec.roundtrip]
  exact congrArg (fun c : Coord view => scratch c.1.val (c.2.val-h.slice)) same.symm

theorem frame {α : Type} (codec : Codec α) (h : Call) (view : View)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Words α) (owner address : Nat)
    (outside : ¬∃ i : Coord view,Active h view indices i ∧ owner=view.owner ∧
      address=view.address i ⟨0,by decide⟩) :
    copy codec h view indices scratch before owner address=before owner address := by
  simp only [copy,dif_neg outside]

theorem inactive_read {α : Type} (codec : Codec α) (h : Call) (view : View)
    (writable : view.Writable) (indices : Nat → Int) (scratch : Nat → Nat → α)
    (before : Words α) (i : Coord view) (inactive : ¬Active h view indices i) :
    read codec view (copy codec h view indices scratch before) i=read codec view before i := by
  unfold read
  congr 1
  apply frame
  rintro ⟨j,active,_,same⟩
  have equal := (byte_injective view writable i j ⟨0,by decide⟩ ⟨0,by decide⟩ same).1
  exact inactive (equal.symm ▸ active)


def realCodec : Codec ℝ where
  encode := fun flag x => if flag then -x else x
  decode := fun flag x => if flag then -x else x
  roundtrip := by intro flag x; cases flag <;> simp

#axiomsClean inactive_read
#axiomsClean readback
#axiomsClean frame
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.LogicalReturn
