import VeriTile.Triton.Float.InputBindingIO
import Mathlib.Tactic.FinCases

/-! Exact-cell inputs preserve dtype requirements without changing decoded-read interfaces. -/

namespace VeriTile.Triton.FP.Structural.InputBindingRegression
noncomputable section

def metadataIO (active : Prop) : ProgramIO Nat Nat where
  inputs := [⟨.int, [], "rows"⟩, ⟨.nat, [], "cache"⟩]
  outputs := []
  scratch := []
  buffers := ["rows", "cache"]
  read := fun _ _ _ => 0
  inputMask := fun _ _ _ => active
  write := fun i => Fin.elim0 i
  outputMask := fun i => Fin.elim0 i
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => []
  encode := fun x => x + 100
  observe := fun i => Fin.elim0 i

def values (active : Prop) : (metadataIO active).Inputs
  | ⟨0, _⟩ => fun _ => (0 : Int)
  | ⟨1, _⟩ => fun _ => (7 : Nat)
  | ⟨i+2,h⟩ => False.elim (by simp [metadataIO] at h)

def mismatched : State Nat where
  mem := fun _ _ => Cell.mk .nat 7
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def tagged : State Nat := { mismatched with mem := fun r _ =>
  if r = "rows" then (Cell.mk .int (0 : Int)) else (Cell.mk .nat (7 : Nat)) }

/-- The existing decoded-read interface admits wrong-tag integer defaults. -/
theorem decoded_default_admitted :
    (metadataIO True).InputBinding (values True) mismatched := by
  intro i j _
  fin_cases i <;> rfl

/-- Exact-cell mode rejects that same memory, rather than silently broadening
the old typed metadata premise to a default-value read. -/
theorem exact_tag_required :
    ¬ (metadataIO True).InputCells (values True) mismatched := by
  intro inputs
  have wrong := inputs (0 : Fin 2) PUnit.unit trivial
  simp [metadataIO,values,mismatched] at wrong

/-- Integer and natural metadata are not transformed by the floating encoder. -/
theorem exact_discrete_payloads_preserved :
    (metadataIO True).InputCells (values True) tagged := by
  intro i j _
  fin_cases i
  · change (if ("rows" : RegionName) = "rows" then (Cell.mk .int (0 : Int))
      else (Cell.mk .nat (7 : Nat))) = (Cell.mk (α := Nat) .int 0)
    simp
  · change (if ("cache" : RegionName) = "rows" then (Cell.mk .int (0 : Int))
      else (Cell.mk .nat (7 : Nat))) = (Cell.mk (α := Nat) .nat 7)
    simp

/-- No dtype or payload requirement leaks into inactive input cells. -/
theorem inactive_cells_unrestricted :
    (metadataIO False).InputCells (values False) mismatched := by
  intro i j active
  exact False.elim active

end
end VeriTile.Triton.FP.Structural.InputBindingRegression
