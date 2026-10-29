import VeriTile.Triton.Float.ProgramIO

/-! Source-connected IO for an ordinary independent-program grid.

Every listed invocation executes from the same initial state. Geometric write
footprints must contain its actual changes and be pairwise disjoint; the final
memory reads those actual source results and preserves every other raw cell.
This retains the established ordinary-grid semantics without replacing it by
serial execution. Footprints receive logical data and coordinates only; no
execution, result, state-transform or safety callback is accepted.
-/
namespace VeriTile.Triton.FP.Structural
noncomputable section

/-- Concrete executions and their disjoint merge. Write sets are geometry,
not a replacement for proving what each source invocation actually changes. -/
structure OrdinaryRun {β : Type} [Inhabited β] (jobs : List (Invocation β))
    (writes : Fin jobs.length → RegionName → Nat → Prop) (initial final : State β) where
  rows : Fin jobs.length → State β
  executed : ∀ i, jobs[i].run initial = some (rows i)
  framed : ∀ i r offset, ¬ writes i r offset → (rows i).mem r offset=initial.mem r offset
  disjoint : ∀ i j r offset, writes i r offset → writes j r offset → i=j
  readback : ∀ i r offset, writes i r offset → final.mem r offset=(rows i).mem r offset
  frame : ∀ r offset, (∀ i, ¬ writes i r offset) → final.mem r offset=initial.mem r offset

namespace OrdinaryRun
open scoped Classical

/-- Canonical disjoint merge reads completed source states, never a formula. -/
def merge {β : Type} {n : Nat} (writes : Fin n → RegionName → Nat → Prop)
    (initial : State β) (rows : Fin n → State β) : State β :=
  { initial with mem := fun r offset =>
      if h : ∃ i, writes i r offset then (rows h.choose).mem r offset
      else initial.mem r offset }

theorem merge_read {β : Type} {n : Nat} (writes : Fin n → RegionName → Nat → Prop)
    (initial : State β) (rows : Fin n → State β)
    (disjoint : ∀ i j r offset, writes i r offset → writes j r offset → i=j)
    (i : Fin n) (r : RegionName) (offset : Nat) (hit : writes i r offset) :
    (merge writes initial rows).mem r offset=(rows i).mem r offset := by
  have present : ∃ j, writes j r offset := ⟨i,hit⟩
  dsimp only [merge]
  rw [dif_pos present,disjoint present.choose i r offset present.choose_spec hit]

theorem merge_frame {β : Type} {n : Nat} (writes : Fin n → RegionName → Nat → Prop)
    (initial : State β) (rows : Fin n → State β) (r : RegionName) (offset : Nat)
    (outside : ∀ i, ¬ writes i r offset) :
    (merge writes initial rows).mem r offset=initial.mem r offset := by
  dsimp only [merge]
  rw [dif_neg (by simpa only [not_exists] using outside)]

/-- Assemble the original grid proof without reproving row execution. -/
def of_rows {β : Type} [Inhabited β] (jobs : List (Invocation β))
    (writes : Fin jobs.length → RegionName → Nat → Prop) (initial : State β)
    (rows : Fin jobs.length → State β)
    (executed : ∀ i, jobs[i].run initial=some (rows i))
    (framed : ∀ i r offset, ¬ writes i r offset → (rows i).mem r offset=initial.mem r offset)
    (disjoint : ∀ i j r offset, writes i r offset → writes j r offset → i=j) :
    OrdinaryRun jobs writes initial (merge writes initial rows) :=
  ⟨rows,executed,framed,disjoint,merge_read writes initial rows disjoint,
    merge_frame writes initial rows⟩

theorem nil_memory {β : Type} [Inhabited β] (writes : Fin 0 → RegionName → Nat → Prop)
    (initial final : State β) (run : OrdinaryRun [] writes initial final) : final.mem=initial.mem := by
  funext r offset
  exact run.frame r offset (fun i => Fin.elim0 i)
end OrdinaryRun

/-- Reuse typed IO fields and invocation data. Here `schedule` is the complete
ordinary grid, not a serial launch sequence. Its execution is `OrdinaryRun`. -/
structure OrdinaryProgramIO (α β : Type) extends ProgramIO α β where
  writes : (vals : ChannelValues α inputs) → Fin (schedule vals).length → RegionName → Nat → Prop

namespace OrdinaryProgramIO
abbrev Inputs {α β : Type} (io : OrdinaryProgramIO α β) := io.toProgramIO.Inputs
abbrev Outputs {α β : Type} (io : OrdinaryProgramIO α β) := io.toProgramIO.Outputs
abbrev InBounds {α β : Type} (io : OrdinaryProgramIO α β) := io.toProgramIO.InBounds
abbrev InputBinding {α β : Type} [Inhabited β] (io : OrdinaryProgramIO α β) := io.toProgramIO.InputBinding
abbrev OutputBinding {α β : Type} (io : OrdinaryProgramIO α β) := io.toProgramIO.OutputBinding
abbrev Frame {α β : Type} (io : OrdinaryProgramIO α β) := io.toProgramIO.Frame

/-- Access safety is about every actual source invocation at the same initial
state. It is a conclusion derived from the declared capacities. -/
def Safe {α β : Type} [Inhabited β] (io : OrdinaryProgramIO α β) (vals : io.Inputs)
    (bounds : RegionName → Nat) (initial : State β) : Prop :=
  ∀ i : Fin (io.schedule vals).length, (io.schedule vals)[i].Safe bounds initial

/-- Source success, typed outputs and exact frame hold independently of backing
capacities; capacities additionally establish safety of those same row runs. -/
def ImplementsOn {α β : Type} [Inhabited β] (io : OrdinaryProgramIO α β)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.Outputs) : Prop :=
  ∀ vals, domain vals → ∀ initial, io.InputBinding vals initial →
    ∃ final, Nonempty (OrdinaryRun (io.schedule vals) (io.writes vals) initial final) ∧
      (∀ bounds, io.InBounds bounds vals → io.Safe vals bounds initial) ∧
      io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final

def Implements {α β : Type} [Inhabited β] (io : OrdinaryProgramIO α β)
    (f : io.Inputs → io.Outputs) : Prop := io.ImplementsOn (fun _ => True) f

scoped infix:50 " ⊨ " => Implements

end OrdinaryProgramIO
end
end VeriTile.Triton.FP.Structural
