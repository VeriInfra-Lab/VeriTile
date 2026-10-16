import VeriTile.Triton.Memory.Flatten

/-! Raw-cell memory facts for finite allocation flattening. These statements
include unallocated flat addresses and preserve every runtime storage tag. -/
namespace VeriTile.Triton.FlatAlloc

/-- A region frame transfers to the complete flat memory, including cells
outside all allocated intervals. Pointer-valued cells retain their relocation. -/
theorem flattenState_frame (A : FlatAlloc) (s t : BlockState)
    (written : RegionName → Nat → Prop)
    (hf : ∀ r o, ¬ written r o → t.mem r o = s.mem r o)
    (r : RegionName) (off : Nat)
    (hn : r ≠ A.flat ∨ ∀ region offset, written region offset → off ≠ A.addr region offset) :
    (A.flattenState t).mem r off = (A.flattenState s).mem r off := by
  change (if r = A.flat then A.readFlat t off else MemCell.real 0) =
    (if r = A.flat then A.readFlat s off else MemCell.real 0)
  by_cases hr : r = A.flat
  · subst r
    simp only [if_pos rfl]
    unfold readFlat
    cases hc : A.decode off with
    | none => rfl
    | some p =>
      apply congrArg A.trCell
      apply hf p.1 p.2
      intro hw
      rcases hn with hn | hn
      · exact hn rfl
      · exact hn p.1 p.2 hw (A.decode_sound hc).2.1
  · simp [hr]

end VeriTile.Triton.FlatAlloc
