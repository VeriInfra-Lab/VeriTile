import VeriTile.Triton.Float.AddressedInputsIO
import Mathlib.Tactic.FinCases

/-! Address-dependent floating interpretations preserve source addresses and integer metadata. -/

namespace VeriTile.Triton.FP.Structural.AddressedInputRegression
noncomputable section

inductive Tagged where
  | ignored
  | leaf (region : RegionName) (offset word : Nat)
  deriving DecidableEq
instance : Inhabited Tagged := ⟨.ignored⟩

def io (payloadRegion : RegionName) : AddressedProgramIO Nat Tagged where
  inputs := [⟨.real, [], payloadRegion⟩, ⟨.int, [], "labels"⟩]
  outputs := []
  scratch := []
  buffers := [payloadRegion, "labels"]
  read := fun i _ _ => if i.val = 0 then 17 else 0
  inputMask := fun _ _ _ => True
  write := fun i => Fin.elim0 i
  outputMask := fun i => Fin.elim0 i
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => []
  encodeAt := fun r offset word => if r = "labels" then .ignored else .leaf r offset word
  observe := fun i => Fin.elim0 i

def values (r : RegionName) : (io r).Inputs
  | ⟨0, _⟩ => fun _ => (7 : Nat)
  | ⟨1, _⟩ => fun _ => (-3 : Int)
  | ⟨i+2,h⟩ => False.elim (by simp [io] at h)

def state : State Tagged where
  mem := fun r offset => if r = "labels" then (.mk .int (-3 : Int))
    else (.mk .real (.leaf r offset 7))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- The encoder receives the actual nonzero read offset and region. -/
theorem actual_address_preserved :
    (io "x").InputBinding (values "x") state := by
  intro i j _
  fin_cases i
  · change (if ("x" : RegionName) = "labels" then (Cell.mk .int (-3 : Int))
      else (Cell.mk .real (Tagged.leaf "x" 17 7))).read .real =
        (if ("x" : RegionName) = "labels" then Tagged.ignored else Tagged.leaf "x" 17 7)
    simp [Cell.read]
  · change (Cell.mk (α := Tagged) .int (-3 : Int)).read .int = (-3 : Int)
    rfl

/-- A label-region floating read can retain its existing default/override
meaning while the signed metadata at the same region is still pinned. -/
theorem labels_alias_preserved :
    (io "labels").InputBinding (values "labels") state := by
  intro i j _
  fin_cases i
  · change (Cell.mk (α := Tagged) .int (-3 : Int)).read .real = Tagged.ignored
    rfl
  · change (Cell.mk (α := Tagged) .int (-3 : Int)).read .int = (-3 : Int)
    rfl

/-- A different address label is not accepted merely because its word agrees. -/
theorem wrong_address_rejected :
    ¬ (io "x").InputBinding (values "x")
      { state with mem := fun _ _ => .mk .real (.leaf "x" 18 7) } := by
  intro binding
  have wrong := binding (0 : Fin 2) PUnit.unit trivial
  change Tagged.leaf "x" 18 7 =
    (if ("x" : RegionName) = "labels" then Tagged.ignored else Tagged.leaf "x" 17 7) at wrong
  simp at wrong

end
end VeriTile.Triton.FP.Structural.AddressedInputRegression
