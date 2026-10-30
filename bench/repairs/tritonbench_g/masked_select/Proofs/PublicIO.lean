import bench.repairs.tritonbench_g.masked_select.IO
import bench.repairs.tritonbench_g.masked_select.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.MaskedSelect.PublicIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MaskedSelect TypedCopy Whole StableSelection
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (copyBackend)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000

theorem launch_initial {α : Type} (rest : Algebra α) (p : Parameters)
    (pid : Nat) (s : State α) : (IO.launch rest p pid).initial s = bindProgram p s pid := rfl

theorem schedule_run {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (jobs : List Nat) (s : State α) :
    runSchedule (jobs.map (IO.launch rest p)) s = runGrid rest p jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid jobs ih =>
    simp only [List.map_cons, runSchedule, runGrid]
    change ((FP.Structural.exec (copyBackend _) (kernel p) (bindProgram p s pid)).bind _) = _
    congr 1
    funext next
    exact ih next

theorem configuration_run {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (config : Config) (s : State α) :
    runSchedule (IO.schedule rest p config) s = Kernel.run rest p config s :=
  schedule_run rest {p with width := config.width} _ s

/-- Exact typed public input cells recover the original prepared-input domain.
Only the prefix relation is required; output values are never assumed. -/
theorem input_cells {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (config : Config) (vals : IO.Inputs α p) (prefixes : IO.Prefixes p vals)
    (s : State α) (pins : (IO.selectionIO rest p config).InputCells vals s) :
    KernelInputs p s (IO.payload vals) (IO.mask vals) := by
  constructor
  · intro i hi
    simpa [IO.payload, hi, mapValue] using
      pins (0 : Fin 3) (⟨i, hi⟩, PUnit.unit) trivial
  · intro i hi
    simpa [IO.mask, hi, mapValue] using
      pins (1 : Fin 3) (⟨i, hi⟩, PUnit.unit) trivial
  · intro i hi
    have h := pins (2 : Fin 3) (⟨i, hi⟩, PUnit.unit) trivial
    change s.mem p.prefixSum.cast i = Cell.mk .nat (vals 2 (⟨i, hi⟩, PUnit.unit)) at h
    rw [prefixes ⟨i, hi⟩] at h
    exact h

/-- The full schedule follows actual predecessor states and preserves the
prepared reads at each step, so every dynamic scatter address remains bounded. -/
theorem schedule_safe {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (jobs : List Nat) (input : Nat → α) (mask : Nat → Bool)
    (sep : KernelSeparate p) (bounds : RegionName → Nat)
    (hinput : ∀ i, i < p.size → i < bounds p.input)
    (hselection : ∀ i, i < p.size → i < bounds p.selection.cast)
    (hprefix : ∀ i, i < p.size → i < bounds p.prefixSum.cast)
    (houtput : ∀ i, i < count mask p.size → i < bounds p.output)
    (s : State α) (reads : KernelInputs p s input mask) :
    ScheduleSafe bounds (jobs.map (IO.launch rest p)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons pid jobs ih =>
    constructor
    · change FP.Structural.TraceSafe (copyBackend _) bounds (kernel p) (bindProgram p s pid)
      apply AccessSafety.source_safe _ p _ bounds hinput hselection hprefix
      intro j active
      have pinned := reads_bind p s pid input mask reads
      rw [offset_selected p _ input mask pinned j active]
      have chosen := (active_iff p _ input mask pinned j).mp active
      exact houtput _ (selected_strict mask chosen.1 chosen.2)
    · intro after ran
      have actual : (IO.launch rest p pid).run s = some (stepValue rest p pid s) :=
        (TypedCopy.execution rest .fp32 p (bindProgram p s pid)).property
      rw [actual] at ran
      cases ran
      exact ih _ (step_reads _ p pid s input mask reads sep)

theorem safe {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (config : Config) (vals : IO.Inputs α p) (prefixes : IO.Prefixes p vals)
    (sep : KernelSeparate p) (bounds : RegionName → Nat)
    (covered : (IO.selectionIO rest p config).InBounds bounds vals)
    (s : State α) (pins : (IO.selectionIO rest p config).InputCells vals s) :
    ScheduleSafe bounds (IO.schedule rest p config) s := by
  have reads := input_cells rest p config vals prefixes s pins
  apply schedule_safe rest {p with width := config.width} _ (IO.payload vals) (IO.mask vals)
    sep bounds
  · intro i hi
    exact covered.2.1 (0 : Fin 3) (⟨i, hi⟩, PUnit.unit) trivial
  · intro i hi
    exact covered.2.1 (1 : Fin 3) (⟨i, hi⟩, PUnit.unit) trivial
  · intro i hi
    exact covered.2.1 (2 : Fin 3) (⟨i, hi⟩, PUnit.unit) trivial
  · intro i hi
    exact covered.2.2.1 (0 : Fin 1)
      (⟨i, lt_of_lt_of_le hi (count_le (IO.mask vals) p.size)⟩, PUnit.unit) hi
  · exact ⟨reads.payload, reads.selection, reads.cumulative⟩

theorem frame_of_result {α : Type} (rest : Algebra α) (p : Parameters)
    (config : Config) (vals : IO.Inputs α p) (before after : State α)
    (frame : Kernel.Frame p (IO.mask vals) before after) :
    (IO.selectionIO rest p config).Frame vals before after := by
  intro r o outside
  apply frame r o
  by_cases region : r = p.output
  · right
    by_contra notBound
    have inside : o < count (IO.mask vals) p.size := by omega
    have index := lt_of_lt_of_le inside (count_le (IO.mask vals) p.size)
    exact ((outside.1 (0 : Fin 1) (⟨o, index⟩, PUnit.unit) inside).resolve_left
      (not_not.mpr region)) rfl
  · exact Or.inl region

theorem implements {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (config : Config) (chosen : config ∈ configs) (sep : KernelSeparate p) :
    (IO.selectionIO rest p config).ImplementsOnCells (IO.Prefixes p) (IO.mathematical p) := by
  apply ProgramIO.ImplementsOnCells.intro
  · intro vals prefixes bounds covered initial pins
    exact safe rest p config vals prefixes sep bounds covered initial pins
  · intro vals prefixes _ _ initial pins
    have reads := input_cells rest p config vals prefixes initial pins
    obtain ⟨final, ran, values, frame⟩ :=
      Kernel.complete rest p config chosen (IO.payload vals) (IO.mask vals) initial reads sep
    refine ⟨final, (configuration_run rest p config initial).trans ran, ?_,
      frame_of_result rest p config vals initial final frame⟩
    intro o
    refine Fin.cases ?_ (fun i => i.elim0) o
    intro j active
    have length : j.1.val < (Kernel.mathematical (IO.payload vals) (IO.mask vals) p.size).length := by
      change j.1.val < (select (IO.payload vals) (IO.mask vals) p.size).length
      simpa only [length_select] using active
    have value := values j.1.val length
    rw [List.getElem?_eq_getElem length] at value
    simp only [Option.map_some, Option.some.injEq] at value
    change mapCell id (final.mem p.output j.1.val) = _
    rw [value]
    change Cell.mk .real _ = Cell.mk .real ((IO.selected p vals)[j.1.val]?.getD default)
    change Cell.mk .real _ = Cell.mk .real
      ((Kernel.mathematical (IO.payload vals) (IO.mask vals) p.size)[j.1.val]?.getD default)
    rw [List.getElem?_eq_getElem length]
    rfl

theorem equivalence {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (left right : Config) (hl : left ∈ configs) (hr : right ∈ configs)
    (sep : KernelSeparate p) :
    (IO.selectionIO rest p left).EquivalentOnCells (IO.candidate rest p left right)
      (IO.Prefixes p) :=
  ProgramIO.EquivalentOnCells.of_implementsOnCells
    (implements rest p left hl sep) (implements rest p right hr sep)

#axiomsClean implements
#axiomsClean equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedSelect.PublicIO
