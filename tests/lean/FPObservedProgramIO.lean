import VeriTile.Triton.Float.ObservedProgramIO
import Mathlib.Tactic.FinCases

/-! Native output observation preserves symbolic inputs, storage dtype and signed metadata. -/

namespace VeriTile.Triton.FP.Structural.ObservedOutputRegression
noncomputable section

/-- The second component stands for symbolic provenance not observed natively. -/
abbrev Symbol := Nat × Nat

def io : ProgramIO Symbol Symbol where
  inputs := [⟨.fp32, [], "word"⟩, ⟨.int, [], "index"⟩]
  outputs := [⟨.fp32, [], "word"⟩, ⟨.int, [], "index"⟩]
  scratch := []
  buffers := ["word", "index"]
  read := fun _ _ _ => 0
  inputMask := fun _ _ _ => True
  write := fun _ _ _ => 0
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => []
  encode := id
  observe := fun _ _ => id

def values (word provenance : Nat) : io.Inputs
  | ⟨0, _⟩ => fun _ => (word, provenance)
  | ⟨1, _⟩ => fun _ => (-3 : Int)
  | ⟨i+2, h⟩ => False.elim (by simp [io] at h)

def state (word provenance : Nat) : State Symbol where
  mem := fun r _ => if r = "word" then .mk .fp32 (word, provenance) else .mk .int (-3 : Int)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def wordObservation : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → Symbol → Nat :=
  fun _ _ term => term.1

/-- The original binding admits all provenance values, including non-leaves. -/
theorem arbitrary_symbolic_input (word provenance : Nat) :
    io.InputBinding (values word provenance) (state word provenance) := by
  intro i j _
  fin_cases i
  · change (if ("word" : RegionName) = "word" then Cell.mk .fp32 (word, provenance)
      else Cell.mk .int (-3 : Int)).read .fp32 = (word, provenance)
    simp [Cell.read]
  · change (if ("index" : RegionName) = "word" then Cell.mk .fp32 (word, provenance)
      else Cell.mk .int (-3 : Int)).read .int = (-3 : Int)
    simp [Cell.read]

/-- Native equality may forget provenance while exact symbolic equality does not. -/
theorem observation_forgets_only_provenance :
    io.ObservedOutputBinding wordObservation (values 7 42)
      (io.mapOutputs wordObservation (values 7 0)) (state 7 42) ∧
    ¬ io.OutputBinding (values 7 42) (values 7 0) (state 7 42) := by
  constructor
  · intro o j _
    fin_cases o
    · change mapCell (fun (x : Symbol) => x.1)
        (mapCell id (if ("word" : RegionName) = "word" then Cell.mk .fp32 (7, 42)
          else Cell.mk .int (-3 : Int))) = Cell.mk .fp32 (7 : Nat)
      rfl
    · change mapCell (fun (x : Symbol) => x.1)
        (mapCell id (if ("index" : RegionName) = "word" then Cell.mk .fp32 (7, 42)
          else Cell.mk .int (-3 : Int))) = Cell.mk .int (-3 : Int)
      rfl
  · intro exactBinding
    have impossible := exactBinding (0 : Fin 2) PUnit.unit trivial
    change mapCell id (if ("word" : RegionName) = "word" then Cell.mk .fp32 (7, 42)
      else Cell.mk .int (-3 : Int)) = Cell.mk .fp32 (7, 0) at impossible
    simp [mapCell, mapValue] at impossible

/-- Forgetting provenance does not erase the FP32 storage tag. -/
theorem wrong_dtype_rejected :
    ¬ io.ObservedOutputBinding wordObservation (values 7 42)
      (io.mapOutputs wordObservation (values 7 0))
      { state 7 42 with mem := fun _ _ => .mk .real (7, 42) } := by
  intro binding
  have impossible := binding (0 : Fin 2) PUnit.unit trivial
  change mapCell (fun (x : Symbol) => x.1) (mapCell id (Cell.mk .real (7, 42))) =
    Cell.mk .fp32 (7 : Nat) at impossible
  simp [mapCell, mapValue] at impossible

/-- Signed metadata is not transformed by the floating output observer. -/
theorem wrong_signed_metadata_rejected :
    ¬ io.ObservedOutputBinding wordObservation (values 7 42)
      (io.mapOutputs wordObservation (values 7 0))
      { state 7 42 with mem := fun _ _ => .mk .int (3 : Int) } := by
  intro binding
  have impossible := binding (1 : Fin 2) PUnit.unit trivial
  change mapCell (fun (x : Symbol) => x.1) (mapCell id (Cell.mk .int (3 : Int))) =
    Cell.mk .int (-3 : Int) at impossible
  simp [mapCell, mapValue] at impossible

end
end VeriTile.Triton.FP.Structural.ObservedOutputRegression
