import bench.tritonbench_g.dequantize_matmul.Proofs.KernelResult

namespace VeriTile.Bench.Repairs.DequantizeMatmul.KernelGrid
open VeriTile.Triton FP.Structural KernelSource KernelExecution KernelResult
noncomputable section
open Classical
set_option maxHeartbeats 3000000

def atProgram {α : Type} (s : State α) (pid : Program) : State α :=
  { s with pids := Function.update (Function.update s.pids 0 pid.1) 1 pid.2 }

def run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (version : Bool) : List Program → State α → Option (State α)
  | [], s => some s
  | pid :: tail, s => do
      let next ← exec M p version (atProgram s pid)
      run M p version tail next

def advance {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) (pid : Program) : State α :=
  (execution M p (atProgram s pid)).val

def finalState {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) : List Program → State α → State α
  | [], s => s
  | pid :: tail, s => finalState M p tail (advance M p s pid)

theorem run_finalState {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (order : List Program) (s : State α) :
    run M p Bool.false order s = some (finalState M p order s) := by
  induction order generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    have hs := (execution M p (atProgram s pid)).property
    change exec M p Bool.false (atProgram s pid) = some (advance M p s pid) at hs
    simpa only [run, finalState, hs, Option.bind_some] using ih (advance M p s pid)

def SameInputs {α : Type} (p : Parameters) (initial current : State α) : Prop :=
  (∀ o, current.mem p.b o = initial.mem p.b o) ∧
  (∀ o, current.mem p.scale o = initial.mem p.scale o)

def Separate (p : Parameters) : Prop := p.b ≠ p.output ∧ p.scale ≠ p.output

def reference {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (o : Nat) : Cell α :=
  .mk .real (dequantized M ((s.mem p.b o).read .int)
    ((s.mem p.scale (o % p.N)).read .real))

def Hit (p : Parameters) (order : List Program) (o : Nat) : Prop :=
  ∃ pid ∈ order, ∃ i : Lane p, active p pid i ∧ offset p pid i = o

theorem hit_cons (p : Parameters) (pid : Program) (tail : List Program) (o : Nat) :
    Hit p (pid :: tail) o ↔
      (∃ i : Lane p, active p pid i ∧ offset p pid i = o) ∨ Hit p tail o := by
  simp [Hit, or_and_right, exists_or]

theorem word_reference {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial s : State α)
    (hs : SameInputs p initial s) (pid : Program) (i : Lane p) (hi : active p pid i) :
    Cell.mk .real (word M p (atProgram s pid) i) =
      reference M p initial (offset p pid i) := by
  have hc := (offset_coordinates p pid i hi).2
  simp [word, reference, atProgram, hi, hi.2, hc, hs.1, hs.2]

theorem advance_point {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial s : State α)
    (hs : SameInputs p initial s) (pid : Program) (r : RegionName) (o : Nat) :
    (advance M p s pid).mem r o =
      if r = p.output ∧ ∃ i : Lane p, active p pid i ∧ offset p pid i = o
      then reference M p initial o else s.mem r o := by
  rw [advance, execution_memory]
  by_cases h : r = p.output ∧ ∃ i : Lane p, active p pid i ∧ offset p pid i = o
  · rw [if_pos h]
    obtain ⟨hr, i, hi, he⟩ := h
    subst r
    subst o
    have hi' : active p ((atProgram s pid).pids 0, (atProgram s pid).pids 1) i := by simpa [atProgram] using hi
    have hw := written_word M p (atProgram s pid) i hi'
    have hw' : (written M p (atProgram s pid)).mem p.output
        (offset p pid i) = Cell.mk .real (word M p (atProgram s pid) i) := by
      simpa [atProgram] using hw
    exact hw'.trans (word_reference M p initial s hs pid i hi)
  · rw [if_neg h]
    apply written_frame
    by_cases hr : r = p.output
    · right
      intro i hi he
      apply h
      refine ⟨hr, i, ?_, ?_⟩
      · simpa [atProgram] using hi
      · simpa [atProgram] using he.symm
    · exact Or.inl hr

theorem advance_inputs {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (sep : Separate p)
    (initial s : State α) (hs : SameInputs p initial s) (pid : Program) :
    SameInputs p initial (advance M p s pid) := by
  constructor
  · intro o
    rw [advance_point M p initial s hs pid]
    have hn : (p.b : RegionName) ≠ p.output := by
      intro h; exact sep.1 h
    simp [hn, hs.1]
  · intro o
    rw [advance_point M p initial s hs pid]
    have hn : (p.scale : RegionName) ≠ p.output := by
      intro h; exact sep.2 h
    simp [hn, hs.2]

theorem final_point {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (sep : Separate p)
    (initial s : State α) (hs : SameInputs p initial s)
    (order : List Program) (r : RegionName) (o : Nat) :
    (finalState M p order s).mem r o =
      if r = p.output ∧ Hit p order o then reference M p initial o else s.mem r o := by
  induction order generalizing s with
  | nil => simp [finalState, Hit]
  | cons pid tail ih =>
    rw [finalState, ih _ (advance_inputs M p sep initial s hs pid),
      advance_point M p initial s hs pid]
    simp only [hit_cons]
    by_cases hr : r = p.output
    · by_cases hh : ∃ i : Lane p, active p pid i ∧ offset p pid i = o
      · simp [hr, hh]
      · simp [hr, hh]
    · simp [hr]

def gridCount (n block : Nat) := (n + block - 1) / block

def order (p : Parameters) : List Program :=
  (List.range (gridCount p.K p.BK)).product (List.range (gridCount p.N p.BN))

theorem quotient_grid (k n block : Nat) (hk : k < n) (hb : 0 < block) :
    k / block < gridCount n block := by
  have bound := Nat.div_mul_le_self k block
  have hm : (k / block + 1) * block ≤ n + block - 1 := by
    rw [Nat.add_mul, Nat.one_mul]
    omega
  have := (Nat.le_div_iff_mul_le hb).mpr hm
  unfold gridCount
  omega

theorem covered (p : Parameters) (hK : 0 < p.BK) (hN : 0 < p.BN)
    (o : Nat) (ho : o < p.K * p.N) : Hit p (order p) o := by
  have hn : 0 < p.N := by nlinarith [Nat.zero_le p.K]
  have hk : o / p.N < p.K := (Nat.div_lt_iff_lt_mul hn).mpr ho
  have hc : o % p.N < p.N := Nat.mod_lt _ hn
  let pid : Program := ((o / p.N) / p.BK, (o % p.N) / p.BN)
  let i : Lane p := (⟨(o / p.N) % p.BK, Nat.mod_lt _ hK⟩,
    ⟨(o % p.N) % p.BN, Nat.mod_lt _ hN⟩, PUnit.unit)
  have hr : row p pid i = o / p.N := by
    simpa [row, pid, i, Nat.mul_comm, Nat.add_comm] using Nat.mod_add_div (o / p.N) p.BK
  have hcol : col p pid i = o % p.N := by
    simpa [col, pid, i, Nat.mul_comm, Nat.add_comm] using Nat.mod_add_div (o % p.N) p.BN
  refine ⟨pid, ?_, i, ?_, ?_⟩
  · exact List.mem_product.mpr
      ⟨List.mem_range.mpr (quotient_grid _ _ _ hk hK),
       List.mem_range.mpr (quotient_grid _ _ _ hc hN)⟩
  · exact ⟨by simpa [hr] using hk, by simpa [hcol] using hc⟩
  · simpa [offset, hr, hcol, Nat.mul_comm, Nat.add_comm] using Nat.mod_add_div o p.N

/-- Active output coordinates identify both the program and the tile lane. -/
theorem unique_owner (p : Parameters) (hK : 0 < p.BK) (hN : 0 < p.BN)
    (pid other : Program) (i j : Lane p) (hi : active p pid i) (hj : active p other j)
    (same : offset p pid i = offset p other j) : pid = other ∧ i = j := by
  have eri := offset_coordinates p pid i hi
  have erj := offset_coordinates p other j hj
  have er := congrArg (· / p.N) same
  have ec := congrArg (· % p.N) same
  dsimp only at er ec
  rw [eri.1, erj.1] at er
  rw [eri.2, erj.2] at ec
  have pr := congrArg (· / p.BK) er
  have pc := congrArg (· / p.BN) ec
  have ownerRow (pid : Program) (i : Lane p) : row p pid i / p.BK = pid.1 := by
    unfold row
    rw [Nat.mul_comm, Nat.mul_add_div hK, Nat.div_eq_of_lt i.1.isLt, Nat.add_zero]
  have ownerCol (pid : Program) (i : Lane p) : col p pid i / p.BN = pid.2 := by
    unfold col
    rw [Nat.mul_comm, Nat.mul_add_div hN, Nat.div_eq_of_lt i.2.1.isLt, Nat.add_zero]
  dsimp only at pr pc
  rw [ownerRow, ownerRow] at pr
  rw [ownerCol, ownerCol] at pc
  have hp : pid = other := Prod.ext pr pc
  subst other
  exact ⟨rfl, offset_injective p pid i j hi hj same⟩

theorem correctness {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (hK : 0 < p.BK) (hN : 0 < p.BN) (sep : Separate p) (s : State α) :
    ∃ final,
      run M p Bool.false
        (order p) s = some final ∧
      (∀ o, o < p.K * p.N → final.mem p.output o = reference M p s o) ∧
      (∀ r o, r ≠ p.output ∨ p.K * p.N ≤ o → final.mem r o = s.mem r o) := by
  refine ⟨finalState M p (order p) s, run_finalState _ _ _ _, ?_, ?_⟩
  · intro o ho
    rw [final_point M p sep s s ⟨fun _ => rfl, fun _ => rfl⟩]
    simp [covered p hK hN o ho]
  · intro r o ho
    rw [final_point M p sep s s ⟨fun _ => rfl, fun _ => rfl⟩]
    rcases ho with hr | hi
    · simp [hr]
    · have hn : ¬ Hit p
          (order p) o := by
        rintro ⟨pid, hp, i, ha, he⟩
        have hh := offset_bound p pid i ha
        omega
      simp [hn]

#axiomsClean run_finalState
#axiomsClean unique_owner
#axiomsClean correctness
#specNonCircular reference avoiding [execution, run, finalState, kernel]
end
end VeriTile.Bench.Repairs.DequantizeMatmul.KernelGrid
