import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.KernelResult

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.KernelGrid
open VeriTile.Triton FP.Structural KernelSource KernelExecution KernelResult
noncomputable section
open Classical
set_option maxHeartbeats 3000000

def atProgram {α : Type} (s : State α) (pid : Nat) : State α :=
  { s with pids := Function.update s.pids 0 pid }

def run {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64)
    (p : Parameters) (version : Bool) : List Nat → State α → Option (State α)
  | [], s => some s
  | pid :: tail, s => do
      let next ← exec M round p version (atProgram s pid)
      run M round p version tail next

def advance {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64)
    (p : Parameters) (s : State α) (pid : Nat) : State α :=
  (execution M round p (atProgram s pid)).val

def finalState {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64)
    (p : Parameters) : List Nat → State α → State α
  | [], s => s
  | pid :: tail, s => finalState M round p tail (advance M round p s pid)

theorem run_finalState {α : Type} [Inhabited α] (M : Algebra α)
    (round : α → BitVec 64) (p : Parameters) (order : List Nat) (s : State α) :
    run M round p Bool.false order s = some (finalState M round p order s) := by
  induction order generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    have hs := (execution M round p (atProgram s pid)).property
    change exec M round p Bool.false (atProgram s pid) = some (advance M round p s pid) at hs
    simpa only [run, finalState, hs, Option.bind_some] using ih (advance M round p s pid)

theorem same_run {α : Type} [Inhabited α] (M : Algebra α)
    (round : α → BitVec 64) (p : Parameters) (order : List Nat) (s : State α) :
    run M round p Bool.false order s = run M round p Bool.true order s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    change (exec M round p Bool.false (atProgram s pid)).bind
        (run M round p Bool.false tail) =
      (exec M round p Bool.true (atProgram s pid)).bind
        (run M round p Bool.true tail)
    rw [KernelExecution.same_run]
    exact congrArg
      (fun (continuation : State α → Option (State α)) =>
        (exec M round p Bool.true (atProgram s pid)).bind continuation)
      (funext fun next => ih next)

def SameInputs {α : Type} (p : Parameters) (initial current : State α) : Prop :=
  (∀ o, current.mem ⟨p.x⟩ o = initial.mem ⟨p.x⟩ o) ∧
  (∀ o, current.mem ⟨p.scale⟩ o = initial.mem ⟨p.scale⟩ o)

def Separate (p : Parameters) : Prop := p.x ≠ p.output ∧ p.scale ≠ p.output

def reference {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64)
    (p : Parameters) (s : State α) (o : Nat) : Cell α :=
  .mk .int (BitVec.ofInt 8 (rounded M round
    ((s.mem ⟨p.x⟩ ((o % p.M) * p.sam + (o / p.M) * p.san)).read .real)
    ((s.mem ⟨p.scale⟩ 0).read .real)).toInt).toInt

def Hit (p : Parameters) (order : List Nat) (o : Nat) : Prop :=
  ∃ pid ∈ order, ∃ i : Lane, active p pid i ∧ outputOffset p pid i = o

theorem hit_cons (p : Parameters) (pid : Nat) (tail : List Nat) (o : Nat) :
    Hit p (pid :: tail) o ↔
      (∃ i : Lane, active p pid i ∧ outputOffset p pid i = o) ∨ Hit p tail o := by
  simp [Hit, or_and_right, exists_or]

theorem word_reference {α : Type} [Inhabited α] (M : Algebra α)
    (round : α → BitVec 64) (p : Parameters) (initial s : State α)
    (hs : SameInputs p initial s) (pid : Nat) (i : Lane) (hi : active p pid i) :
    Cell.mk .int (BitVec.ofInt 8 (word M round p (atProgram s pid) i).toInt).toInt =
      reference M round p initial (outputOffset p pid i) := by
  have hd := QuantTransposeFinite.encode_decode p.M (row p pid i) (col p pid i) hi.1
  have hr : outputOffset p pid i % p.M = row p pid i := by
    simpa [outputOffset, Nat.add_comm] using hd.1
  have hc : outputOffset p pid i / p.M = col p pid i := by
    simpa [outputOffset, Nat.add_comm] using hd.2
  simp [word, reference, atProgram, hi, hr, hc, inputOffset, hs.1, hs.2]

theorem advance_point {α : Type} [Inhabited α] (M : Algebra α)
    (round : α → BitVec 64) (p : Parameters) (initial s : State α)
    (hs : SameInputs p initial s) (pid : Nat) (r : RegionName) (o : Nat) :
    (advance M round p s pid).mem r o =
      if r = ⟨p.output⟩ ∧ ∃ i : Lane, active p pid i ∧ outputOffset p pid i = o
      then reference M round p initial o else s.mem r o := by
  rw [advance, execution_memory]
  by_cases h : r = ⟨p.output⟩ ∧ ∃ i : Lane, active p pid i ∧ outputOffset p pid i = o
  · rw [if_pos h]
    obtain ⟨hr, i, hi, he⟩ := h
    subst r
    subst o
    have hi' : active p ((atProgram s pid).pids 0) i := by simpa [atProgram] using hi
    have hw := written_word M round p (atProgram s pid) i hi'
    have hw' : (written M round p (atProgram s pid)).mem ⟨p.output⟩
        (outputOffset p pid i) = Cell.mk .int (BitVec.ofInt 8 (word M round p (atProgram s pid) i).toInt).toInt := by
      simpa only [atProgram, Function.update_self] using hw
    exact hw'.trans (word_reference M round p initial s hs pid i hi)
  · rw [if_neg h]
    apply written_frame
    by_cases hr : r = ⟨p.output⟩
    · right
      intro i hi he
      apply h
      refine ⟨hr, i, ?_, ?_⟩
      · simpa [atProgram] using hi
      · simpa [atProgram] using he.symm
    · exact Or.inl hr

theorem advance_inputs {α : Type} [Inhabited α] (M : Algebra α)
    (round : α → BitVec 64) (p : Parameters) (sep : Separate p)
    (initial s : State α) (hs : SameInputs p initial s) (pid : Nat) :
    SameInputs p initial (advance M round p s pid) := by
  constructor
  · intro o
    rw [advance_point M round p initial s hs pid]
    have hn : (⟨p.x⟩ : RegionName) ≠ ⟨p.output⟩ := by
      intro h; exact sep.1 (congrArg Region.name h)
    simp [hn, hs.1]
  · intro o
    rw [advance_point M round p initial s hs pid]
    have hn : (⟨p.scale⟩ : RegionName) ≠ ⟨p.output⟩ := by
      intro h; exact sep.2 (congrArg Region.name h)
    simp [hn, hs.2]

theorem final_point {α : Type} [Inhabited α] (M : Algebra α)
    (round : α → BitVec 64) (p : Parameters) (sep : Separate p)
    (initial s : State α) (hs : SameInputs p initial s)
    (order : List Nat) (r : RegionName) (o : Nat) :
    (finalState M round p order s).mem r o =
      if r = ⟨p.output⟩ ∧ Hit p order o then reference M round p initial o else s.mem r o := by
  induction order generalizing s with
  | nil => simp [finalState, Hit]
  | cons pid tail ih =>
    rw [finalState, ih _ (advance_inputs M round p sep initial s hs pid),
      advance_point M round p initial s hs pid]
    simp only [hit_cons]
    by_cases hr : r = ⟨p.output⟩
    · by_cases hh : ∃ i : Lane, active p pid i ∧ outputOffset p pid i = o
      · simp [hr, hh]
      · simp [hr, hh]
    · simp [hr]

def laneOfFlat (i : Nat) (hi : i < 128 * 128) : Lane :=
  (⟨i / 128, by omega⟩, ⟨i % 128, Nat.mod_lt _ (by decide)⟩, PUnit.unit)

theorem flat_laneOfFlat (i : Nat) (hi : i < 128 * 128) : flat (laneOfFlat i hi) = i := by
  dsimp [flat, laneOfFlat]
  omega

theorem covered (p : Parameters) (o : Nat) (ho : o < p.M * p.N) :
    Hit p (List.range (QuantTransposeFinite.gridM p.M * QuantTransposeFinite.gridN p.N)) o := by
  obtain ⟨hr, hc, he⟩ := QuantTransposeFinite.decode_bounds p.M p.N o ho
  obtain ⟨pid, hp, lane, hl, hm, hn⟩ :=
    QuantTransposeFinite.coordinate_preimage p.M p.N (o % p.M) (o / p.M) hr hc
  let i := laneOfFlat lane hl
  have hm' : row p pid i = o % p.M := by
    rw [← flat_row, flat_laneOfFlat]; exact hm
  have hn' : col p pid i = o / p.M := by
    rw [← flat_col, flat_laneOfFlat]; exact hn
  exact ⟨pid, List.mem_range.mpr hp, i, ⟨by simpa [active, hm', hn'] using And.intro hr hc,
    by simpa [outputOffset, hm', hn', Nat.add_comm] using he⟩⟩

theorem correctness {α : Type} [Inhabited α] (M : Algebra α)
    (round : α → BitVec 64) (p : Parameters) (sep : Separate p) (s : State α) :
    ∃ final,
      run M round p Bool.false
        (List.range (QuantTransposeFinite.gridM p.M * QuantTransposeFinite.gridN p.N)) s = some final ∧
      (∀ o, o < p.M * p.N → final.mem ⟨p.output⟩ o = reference M round p s o) ∧
      (∀ r o, r ≠ ⟨p.output⟩ ∨ p.M * p.N ≤ o → final.mem r o = s.mem r o) := by
  refine ⟨finalState M round p (List.range _) s, run_finalState _ _ _ _ _, ?_, ?_⟩
  · intro o ho
    rw [final_point M round p sep s s ⟨fun _ => rfl, fun _ => rfl⟩]
    simp [covered p o ho]
  · intro r o ho
    rw [final_point M round p sep s s ⟨fun _ => rfl, fun _ => rfl⟩]
    rcases ho with hr | hi
    · simp [hr]
    · have hn : ¬ Hit p
          (List.range (QuantTransposeFinite.gridM p.M * QuantTransposeFinite.gridN p.N)) o := by
        rintro ⟨pid, hp, i, ha, he⟩
        have hh := output_bound p pid i ha
        omega
      simp [hn]

#axiomsClean run_finalState
#axiomsClean same_run
#axiomsClean correctness
#specNonCircular reference avoiding [execution, run, finalState, kernel]
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.KernelGrid
