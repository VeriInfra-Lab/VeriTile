import bench.tritonbench_g.masked_select.Proofs.HostSource

namespace VeriTile.Bench.TritonBenchG.MaskedSelect.Whole
open VeriTile.Triton FP.Structural TypedCopy StableSelection
noncomputable section
open Classical
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

structure Reads {α : Type} (p : Parameters) (s : State α)
    (input : Nat → α) (mask : Nat → Bool) : Prop where
  payload : ∀ i, i < p.size → s.mem p.input i = .mk .real (input i)
  selection : ∀ i, i < p.size → s.mem p.selection.cast i = .mk .bool (mask i)
  cumulative : ∀ i, i < p.size → s.mem p.prefixSum.cast i = .mk .nat (count mask (i + 1))

def Separate (p : Parameters) : Prop :=
  p.input ≠ p.output ∧ p.selection.cast ≠ p.output ∧ p.prefixSum.cast ≠ p.output

def stepValue {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (pid : Nat) (s : State α) : State α :=
  (TypedCopy.execution rest .fp32 p (bindProgram p s pid)).val

def gridValue {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters) :
    List Nat → State α → State α
  | [], s => s
  | pid :: jobs, s => gridValue rest p jobs (stepValue rest p pid s)

theorem runGrid_value {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (jobs : List Nat) (s : State α) : runGrid rest p jobs s = some (gridValue rest p jobs s) := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid jobs ih =>
    unfold runGrid
    rw [(TypedCopy.execution rest .fp32 p (bindProgram p s pid)).property]
    exact ih _

theorem reads_bind {α : Type} (p : Parameters) (s : State α) (pid : Nat)
    (input : Nat → α) (mask : Nat → Bool) (h : Reads p s input mask) :
    Reads p (bindProgram p s pid) input mask :=
  ⟨h.payload, h.selection, h.cumulative⟩

theorem active_iff {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (input : Nat → α) (mask : Nat → Bool) (h : Reads p s input mask)
    (i : TileIndex [p.width]) :
    TypedCopy.active p s i ↔ mask (inputOffset p s i) = Bool.true ∧ inputOffset p s i < p.size := by
  constructor
  · rintro ⟨hm, hn⟩
    rw [h.selection _ hn, Cell.read_same] at hm
    exact ⟨hm, hn⟩
  · rintro ⟨hm, hn⟩
    exact ⟨by rw [h.selection _ hn, Cell.read_same]; exact hm, hn⟩

theorem offset_selected {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (input : Nat → α) (mask : Nat → Bool) (h : Reads p s input mask)
    (i : TileIndex [p.width]) (hi : TypedCopy.active p s i) :
    outputOffset p s i = count mask (inputOffset p s i) := by
  have ha := (active_iff p s input mask h i).mp hi
  unfold outputOffset
  rw [h.cumulative _ ha.2, Cell.read_same]
  exact (inclusive_selected mask ha.1).1

theorem prefix_injective {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (input : Nat → α) (mask : Nat → Bool) (h : Reads p s input mask) :
    ∀ a b, TypedCopy.active p s a → TypedCopy.active p s b → outputOffset p s a = outputOffset p s b → a = b := by
  intro a b ha hb heq
  rw [offset_selected p s input mask h a ha, offset_selected p s input mask h b hb] at heq
  have hidx := selected_injective mask ((active_iff p s input mask h a).mp ha).1
    ((active_iff p s input mask h b).mp hb).1 heq
  have hval : a.1.val = b.1.val := by unfold inputOffset at hidx; omega
  exact Prod.ext (Fin.ext hval) (Subsingleton.elim _ _)

theorem step_reads {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (pid : Nat) (s : State α) (input : Nat → α) (mask : Nat → Bool)
    (h : Reads p s input mask) (sep : Separate p) :
    Reads p (stepValue rest p pid s) input mask := by
  constructor
  · intro i hi
    rw [stepValue, TypedCopy.frame rest .fp32 p _ p.input i (Or.inl sep.1)]
    exact h.payload i hi
  · intro i hi
    rw [stepValue, TypedCopy.frame rest .fp32 p _ p.selection.cast i (Or.inl sep.2.1)]
    exact h.selection i hi
  · intro i hi
    rw [stepValue, TypedCopy.frame rest .fp32 p _ p.prefixSum.cast i (Or.inl sep.2.2)]
    exact h.cumulative i hi

theorem lane_owner {α : Type} (p : Parameters) (s : State α) (pid : Nat)
    (hwidth : 0 < p.width) (lane : TileIndex [p.width]) :
    inputOffset p (bindProgram p s pid) lane / p.width = pid := by
  simp only [inputOffset, bindProgram, ↓reduceIte]
  rw [Nat.mul_comm pid p.width, Nat.mul_add_div hwidth]
  simp [Nat.div_eq_of_lt lane.1.isLt]

theorem step_selected {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (pid : Nat) (s : State α) (input : Nat → α) (mask : Nat → Bool)
    (h : Reads p s input mask) (hwidth : 0 < p.width)
    (i : Nat) (hi : i < p.size) (hm : mask i = Bool.true) :
    (stepValue rest p pid s).mem p.output (count mask i) =
      if i / p.width = pid then .mk .real (input i) else s.mem p.output (count mask i) := by
  have hr := reads_bind p s pid input mask h
  by_cases hp : i / p.width = pid
  · rw [if_pos hp]
    let lane : TileIndex [p.width] := (⟨i % p.width, Nat.mod_lt i hwidth⟩, PUnit.unit)
    have hoff : inputOffset p (bindProgram p s pid) lane = i := by
      simp only [inputOffset, bindProgram, ↓reduceIte, lane]
      rw [← hp, Nat.mul_comm, Nat.div_add_mod]
    have ha : TypedCopy.active p (bindProgram p s pid) lane := by
      rw [active_iff p _ input mask hr, hoff]
      exact ⟨hm, hi⟩
    have hout := offset_selected p _ input mask hr lane ha
    rw [hoff] at hout
    rw [stepValue, ← hout, TypedCopy.typed_readback rest .fp32 p _ lane ha
      (prefix_injective p _ input mask hr) (input i) (by simpa [hoff] using hr.payload i hi)]
    rfl
  · rw [if_neg hp]
    apply TypedCopy.frame rest .fp32 p (bindProgram p s pid) p.output (count mask i)
    right
    intro lane ha heq
    have hl := (active_iff p _ input mask hr lane).mp ha
    rw [offset_selected p _ input mask hr lane ha] at heq
    have hid := selected_injective mask hm hl.1 heq
    have own := lane_owner p s pid hwidth lane
    rw [← hid] at own
    exact hp own

theorem step_frame {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (pid : Nat) (s : State α) (input : Nat → α) (mask : Nat → Bool)
    (h : Reads p s input mask) (r : RegionName) (o : Nat)
    (outside : r ≠ p.output ∨ count mask p.size ≤ o) :
    (stepValue rest p pid s).mem r o = s.mem r o := by
  apply TypedCopy.frame rest .fp32 p (bindProgram p s pid) r o
  rcases outside with hr | ho
  · exact Or.inl hr
  · right
    intro lane ha
    have hreads := reads_bind p s pid input mask h
    have hi := (active_iff p _ input mask hreads lane).mp ha
    rw [offset_selected p _ input mask hreads lane ha]
    have := selected_strict mask hi.1 hi.2
    omega

theorem grid_reads {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (jobs : List Nat) (s : State α) (input : Nat → α) (mask : Nat → Bool)
    (h : Reads p s input mask) (sep : Separate p) :
    Reads p (gridValue rest p jobs s) input mask := by
  induction jobs generalizing s with
  | nil => exact h
  | cons pid jobs ih => exact ih _ (step_reads rest p pid s input mask h sep)

theorem grid_selected {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (jobs : List Nat) (s : State α) (input : Nat → α) (mask : Nat → Bool)
    (h : Reads p s input mask) (sep : Separate p) (hwidth : 0 < p.width)
    (i : Nat) (hi : i < p.size) (hm : mask i = Bool.true) :
    (gridValue rest p jobs s).mem p.output (count mask i) =
      if i / p.width ∈ jobs then .mk .real (input i) else s.mem p.output (count mask i) := by
  induction jobs generalizing s with
  | nil => simp [gridValue]
  | cons pid jobs ih =>
    rw [gridValue, ih _ (step_reads rest p pid s input mask h sep)]
    rw [step_selected rest p pid s input mask h hwidth i hi hm]
    by_cases ht : i / p.width ∈ jobs <;> by_cases hh : i / p.width = pid <;> simp [ht, hh]

theorem grid_frame {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (jobs : List Nat) (s : State α) (input : Nat → α) (mask : Nat → Bool)
    (h : Reads p s input mask) (sep : Separate p) (r : RegionName) (o : Nat)
    (outside : r ≠ p.output ∨ count mask p.size ≤ o) :
    (gridValue rest p jobs s).mem r o = s.mem r o := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid jobs ih =>
    rw [gridValue, ih _ (step_reads rest p pid s input mask h sep)]
    exact step_frame rest p pid s input mask h r o outside

theorem grid_covers (n width i : Nat) (hwidth : 0 < width) (hi : i < n) :
    i / width < grid n width := by
  apply (Nat.div_lt_iff_lt_mul hwidth).mpr
  have hd := Nat.div_add_mod (n + width - 1) width
  have hm := Nat.mod_lt (n + width - 1) hwidth
  rw [Nat.mul_comm] at hd
  unfold grid
  omega

/-- Full scheduled execution, from produced inputs to independent stable
selection. Completeness requires each launched pid, but permits extra repeats
(autotuning) and any order; the public native domain separately bounds pids. -/
theorem complete_grid {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (jobs : List Nat) (s : State α) (input : Nat → α) (mask : Nat → Bool)
    (h : Reads p s input mask) (sep : Separate p) (hwidth : 0 < p.width)
    (complete : ∀ pid, pid < grid p.size p.width → pid ∈ jobs) :
    ∃ t, runGrid rest p jobs s = some t ∧ Reads p t input mask ∧
      (∀ k, k < (select input mask p.size).length →
        some (t.mem p.output k) = ((select input mask p.size)[k]?).map (Cell.mk .real)) ∧
      (∀ r o, r ≠ p.output ∨ count mask p.size ≤ o → t.mem r o = s.mem r o) := by
  refine ⟨gridValue rest p jobs s, runGrid_value rest p jobs s,
    grid_reads rest p jobs s input mask h sep, ?_, grid_frame rest p jobs s input mask h sep⟩
  intro k hk
  rw [length_select] at hk
  obtain ⟨i, hi, hm, hc⟩ := selected_surjective mask hk
  rw [← hc, get_selected input mask hi hm]
  simp only [Option.map_some]
  rw [grid_selected rest p jobs s input mask h sep hwidth i hi hm,
    if_pos (complete _ (grid_covers p.size p.width i hwidth hi))]

#axiomsClean complete_grid
#axiomsClean prefix_injective
#axiomsClean grid_frame
end
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.MaskedSelect.Whole
