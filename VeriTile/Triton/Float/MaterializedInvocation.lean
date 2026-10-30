import VeriTile.Triton.Float.ProgramIO

/-! Per-CTA materialization of actual Structural source results.

A codec acts on one floating payload at one fixed address; `mapCell` retains
all dtype tags and all discrete payloads. Each launch starts with fresh local
registers and default masked-load fallback values, then uses the ordinary
`Invocation` ABI setup. The next launch receives the decoded actual result
memory. No memory transformer, runner, or safety predicate is configurable.
-/
namespace VeriTile.Triton.FP.Structural
noncomputable section

structure MaterializedInvocation (α β : Type) where
  source : Invocation β
  encode : α → β
  decodeAt : RegionName → Nat → β → α
  decode_encode : ∀ region offset value, decodeAt region offset (encode value) = value

namespace MaterializedInvocation
variable {α β : Type}

/-- Only floating payloads are embedded; integer/address channels keep their
original values and tags. -/
def embed (job : MaterializedInvocation α β) (memory : RegionName → Nat → Cell α) :
    RegionName → Nat → Cell β := fun region offset => mapCell job.encode (memory region offset)

/-- Finish each actual stored scalar at the fixed address's selected owner. -/
def finish (job : MaterializedInvocation α β) (memory : RegionName → Nat → Cell β) :
    RegionName → Nat → Cell α :=
  fun region offset => mapCell (job.decodeAt region offset) (memory region offset)

/-- The selected source model's fixed fresh local state. `Invocation.initial`
subsequently installs its concrete arguments, pids and numPids. -/
def embedded [Inhabited β] (job : MaterializedInvocation α β)
    (memory : RegionName → Nat → Cell α) : State β where
  mem := job.embed memory
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun dtype _ _ => defaultValue dtype

def run [Inhabited β] (job : MaterializedInvocation α β)
    (memory : RegionName → Nat → Cell α) : Option (RegionName → Nat → Cell α) :=
  (job.source.run (job.embedded memory)).map (fun final => job.finish final.mem)

/-- Safety is checked in the same embedded predecessor state as execution. -/
def Safe [Inhabited β] (job : MaterializedInvocation α β) (bounds : RegionName → Nat)
    (memory : RegionName → Nat → Cell α) : Prop := job.source.Safe bounds (job.embedded memory)

@[simp] theorem finish_embed (job : MaterializedInvocation α β)
    (memory : RegionName → Nat → Cell α) : job.finish (job.embed memory) = memory := by
  funext region offset
  simp only [finish,embed]
  cases memory region offset with
  | mk dtype value => cases dtype <;> simp [mapCell,mapValue,job.decode_encode]

/-- An unchanged actual source cell remains exactly unchanged after decoding,
including its tag and any discrete payload. -/
theorem finish_frame (job : MaterializedInvocation α β)
    (before : RegionName → Nat → Cell α) (after : RegionName → Nat → Cell β)
    (region : RegionName) (offset : Nat)
    (unchanged : after region offset = job.embed before region offset) :
    job.finish after region offset = before region offset := by
  change mapCell _ (after region offset) = _
  rw [unchanged]
  exact congrFun (congrFun (job.finish_embed before) region) offset

theorem run_eq [Inhabited β] (job : MaterializedInvocation α β)
    (memory : RegionName → Nat → Cell α) (final : State β)
    (executed : job.source.run (job.embedded memory) = some final) :
    job.run memory = some (job.finish final.mem) := by
  simp only [run,executed,Option.map_some]

end MaterializedInvocation

/-- Every CTA receives the materialized actual memory of its predecessor. -/
def runMaterializedSchedule {α β : Type} [Inhabited β] :
    List (MaterializedInvocation α β) → (RegionName → Nat → Cell α) →
      Option (RegionName → Nat → Cell α)
  | [], memory => some memory
  | job :: rest, memory => (job.run memory).bind (runMaterializedSchedule rest)

/-- All actual reached invocations are checked, even when no output channel is
active. Execution success is separately required by the public IO judgment. -/
def MaterializedScheduleSafe {α β : Type} [Inhabited β] (bounds : RegionName → Nat) :
    List (MaterializedInvocation α β) → (RegionName → Nat → Cell α) → Prop
  | [], _ => True
  | job :: rest, memory => job.Safe bounds memory ∧
      ∀ next, job.run memory = some next → MaterializedScheduleSafe bounds rest next

@[simp] theorem runMaterializedSchedule_singleton {α β : Type} [Inhabited β]
    (job : MaterializedInvocation α β) (memory : RegionName → Nat → Cell α) :
    runMaterializedSchedule [job] memory = job.run memory := by
  cases h : job.run memory <;> simp [runMaterializedSchedule,h]

@[simp] theorem materializedScheduleSafe_singleton {α β : Type} [Inhabited β]
    (bounds : RegionName → Nat) (job : MaterializedInvocation α β)
    (memory : RegionName → Nat → Cell α) :
    MaterializedScheduleSafe bounds [job] memory ↔ job.Safe bounds memory := by
  simp [MaterializedScheduleSafe]

theorem runMaterializedSchedule_append {α β : Type} [Inhabited β]
    (xs ys : List (MaterializedInvocation α β)) (memory : RegionName → Nat → Cell α) :
    runMaterializedSchedule (xs ++ ys) memory =
      (runMaterializedSchedule xs memory).bind (runMaterializedSchedule ys) := by
  induction xs generalizing memory with
  | nil => rfl
  | cons job rest ih =>
    simp only [List.cons_append,runMaterializedSchedule]
    cases h : job.run memory <;> simp [ih]

theorem materializedScheduleSafe_append {α β : Type} [Inhabited β]
    (bounds : RegionName → Nat) (xs ys : List (MaterializedInvocation α β))
    (memory : RegionName → Nat → Cell α) :
    MaterializedScheduleSafe bounds (xs ++ ys) memory ↔
      MaterializedScheduleSafe bounds xs memory ∧
      ∀ next, runMaterializedSchedule xs memory = some next →
        MaterializedScheduleSafe bounds ys next := by
  induction xs generalizing memory with
  | nil => simp [MaterializedScheduleSafe,runMaterializedSchedule]
  | cons job rest ih =>
    simp only [List.cons_append,MaterializedScheduleSafe,runMaterializedSchedule]
    cases h : job.run memory <;> simp [ih,and_assoc]

end
end VeriTile.Triton.FP.Structural
