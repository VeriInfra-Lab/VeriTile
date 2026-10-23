import bench.tritonbench_g.dropout_triton.Proofs.Real
import bench.repairs.tritonbench_g.dropout_triton.Kernels
import bench.tritonbench_g.vector_addition.Proofs.HostGrid
import VeriTile.Triton.Launch.Grid
import VeriTile.Meta.StatementAudit

/-! Complete BLOCK1024 mathematical launch using the unchanged dropout
per-program theorem. All Real inputs and p are symbolic; keep truth is explicit.
This theorem is not an IEEE division or host-runtime claim. -/
namespace VeriTile.Bench.Repairs.DropoutTriton.RealGrid
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DropoutTriton TypedSource
open VeriTile.Bench.TritonBenchG.VectorAddition
noncomputable section
open Classical
set_option maxHeartbeats 4000000

def count (N : Nat) : Nat := (N + 1024 - 1) / 1024
def grid (N : Nat) : Grid := ⟨[count N]⟩
def kernel (keep : KeepStorage) (X K O : RegionName) (N : Nat) (p : ℝ) : Kernel :=
  (RealProof.original keep X K O N 1024 p).toAlgKernel

/-- Only launch metadata changes; memory is the current evolving memory. -/
def programState (s : BlockState) (N pid : Nat) : BlockState :=
  (s.withPids (fun ax => if ax = 0 then pid else 0)).withNumPids
    (fun ax => if ax = 0 then count N else 1)

def runPrograms (keep : KeepStorage) (X K O : RegionName) (N : Nat) (p : ℝ) : List Nat → BlockState → Option BlockState
  | [], s => some s
  | pid :: rest, s => do
      let t ← exec (kernel keep X K O N p) (programState s N pid)
      runPrograms keep X K O N p rest t

def Frame (O : RegionName) (N : Nat) (programs : List Nat) (s t : BlockState) : Prop :=
  ∀ r o, (r ≠ O ∨ N ≤ o ∨ o / 1024 ∉ programs) → t.mem r o = s.mem r o

def Values (keep : KeepStorage) (X K O : RegionName) (N : Nat) (p : ℝ) (programs : List Nat) (s t : BlockState) : Prop :=
  ∀ o, o < N → o / 1024 ∈ programs →
    t.readMem O o = RealProof.expected keep s X K p o

theorem read_eq_of_mem {s t : BlockState} {r : RegionName} {o : Nat}
    (h : t.mem r o = s.mem r o) : t.readMem r o = s.readMem r o := by
  simp only [BlockState.readMem, h]

/-- Reuses the established original DSL execution theorem, including its raw
frame. The input tile is loaded before the output scatter. -/
theorem program_run (keep : KeepStorage) (X K O : RegionName) (N pid : Nat) (p : ℝ) (s : BlockState) :
    ∃ t, exec (kernel keep X K O N p) (programState s N pid) = some t ∧
      Values keep X K O N p [pid] s t ∧ Frame O N [pid] s t := by
  obtain ⟨t, he, hv, hf⟩ := RealProof.run keep X K O N 1024 p (programState s N pid)
  refine ⟨t, he, ?_, ?_⟩
  · intro o ho hp
    have hp' : o / 1024 = pid := by simpa using hp
    let j : Fin 1024 := ⟨o % 1024, Nat.mod_lt _ (by decide)⟩
    have hj : pid * 1024 + j.val = o := by dsimp [j]; omega
    have hh := hv j (by simpa [programState, RealProof.offset, RealProof.expected, RealProof.readKeep, hj] using ho)
    simpa [programState, RealProof.offset, RealProof.expected, RealProof.readKeep, hj] using hh
  · intro r o hout
    apply hf r o
    rcases hout with hr | hn | hp
    · exact Or.inl hr
    · right; intro j hj
      have hj' : pid * 1024 + j.val < N := hj
      omega
    · right; intro j hj heq
      have hpid : o / 1024 = pid := by
        have := j.isLt
        change o = pid * 1024 + j.val at heq
        omega
      exact hp (by simp [hpid])

/-- Distinct program tiles cannot change the input cell of a later tile.
This includes exact logical in-place X=O; the host uses fresh allocation. -/
theorem runPrograms_correct (keep : KeepStorage) (X K O : RegionName) (N : Nat) (p : ℝ) (programs : List Nat)
    (unique : programs.Nodup) (s : BlockState) :
    ∃ t, runPrograms keep X K O N p programs s = some t ∧
      Values keep X K O N p programs s t ∧ Frame O N programs s t := by
  induction programs generalizing s with
  | nil => exact ⟨s, rfl, by simp [Values], by intro r o _; rfl⟩
  | cons pid rest ih =>
    have hp : pid ∉ rest := (List.nodup_cons.mp unique).1
    have hr : rest.Nodup := (List.nodup_cons.mp unique).2
    obtain ⟨mid, hm, hvm, hfm⟩ := program_run keep X K O N pid p s
    obtain ⟨t, ht, hvt, hft⟩ := ih hr mid
    refine ⟨t, by simp only [runPrograms, hm]; exact ht, ?_, ?_⟩
    · intro o ho hin
      rcases List.mem_cons.mp hin with hhead | htail
      · have hrest : o / 1024 ∉ rest := by simpa [hhead] using hp
        have keep := read_eq_of_mem (hft O o (Or.inr (Or.inr hrest)))
        exact keep.trans (hvm o ho (by simp [hhead]))
      · have hnot : o / 1024 ∉ [pid] := by
          simp only [List.mem_singleton]
          intro heq; exact hp (heq ▸ htail)
        have hx := hfm X o (Or.inr (Or.inr hnot))
        have hk := hfm K o (Or.inr (Or.inr hnot))
        rw [hvt o ho htail]
        cases keep <;> simp [RealProof.expected,RealProof.readKeep,BlockState.readMem,
          BlockState.readMemValue,BlockState.readMemTyped,hx,hk]
    · intro r o outside
      have a : r ≠ O ∨ N ≤ o ∨ o / 1024 ∉ rest := by
        rcases outside with h | h | h
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
        · exact Or.inr (Or.inr (fun hx => h (List.mem_cons_of_mem pid hx)))
      have b : r ≠ O ∨ N ≤ o ∨ o / 1024 ∉ [pid] := by
        rcases outside with h | h | h
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
        · right; right; intro hx
          have heq : o / 1024 = pid := by simpa using hx
          exact h (by simp [heq])
      exact (hft r o a).trans (hfm r o b)

/-- Any complete whole-program permutation covers every mathematical input
index and preserves every exterior raw cell, including the masked tail. -/
theorem complete_schedule_correct (keep : KeepStorage) (X K O : RegionName) (N : Nat) (p : ℝ) (programs : List Nat)
    (schedule : programs.Perm (List.range (count N))) (s : BlockState) :
    ∃ t, runPrograms keep X K O N p programs s = some t ∧
      (∀ o, o < N → t.readMem O o = RealProof.expected keep s X K p o) ∧
      (∀ r o, r ≠ O ∨ N ≤ o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hv, hf⟩ := runPrograms_correct keep X K O N p programs
    (schedule.nodup_iff.mpr List.nodup_range) s
  refine ⟨t, he, ?_, ?_⟩
  · intro o ho
    have cover := ValueProofs.grid_covers (⟨X, X, O, N, 1024⟩ : ValueProofs.Parameters)
      (by change 0 < (1024 : Nat); decide)
    change N ≤ count N * 1024 at cover
    have owner : o / 1024 < count N := by omega
    exact hv o ho (schedule.mem_iff.mpr (List.mem_range.mpr owner))
  · intro r o h
    apply hf r o
    rcases h with h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)

theorem full_grid_correct (keep : KeepStorage) (X K O : RegionName) (N : Nat) (p : ℝ) (s : BlockState) :
    ∃ t, runPrograms keep X K O N p (List.range (count N)) s = some t ∧
      (∀ o, o < N → t.readMem O o = RealProof.expected keep s X K p o) ∧
      (∀ r o, r ≠ O ∨ N ≤ o → t.mem r o = s.mem r o) :=
  complete_schedule_correct keep X K O N p _ (List.Perm.refl _) s

theorem empty_grid (keep : KeepStorage) (X K O : RegionName) (p : ℝ) (s : BlockState) :
    runPrograms keep X K O 0 p (List.range (count 0)) s = some s := by
  rfl

/-- Finite signed-index correspondence is an extra layer, not a bound on the
mathematical theorem. The revised wrapper supplies this bound through its
checked InputContract. -/
theorem source_no_wrap_bound (N pid lane : Nat) (hN : N < 2^31)
    (hp : pid < count N) (hl : lane < 1024) : pid * 1024 + lane < 2^31 := by
  simp only [count] at hp
  norm_num at hN ⊢
  omega

def gridPid {N : Nat} (idx : GridIndex (grid N)) : Nat :=
  (idx ⟨0, by simp [grid, Grid.rank]⟩).val

theorem programState_eq_grid {N : Nat} (s : BlockState) (idx : GridIndex (grid N)) :
    programState s N (gridPid idx) = s.withGridIndex idx := by
  apply BlockState.ext
  · intro r o; rfl
  · intro d sh name; rfl
  · intro ax
    by_cases h : ax = 0
    · subst ax; rfl
    · have hh : ¬ax < 1 := by omega
      simp [programState, BlockState.withGridIndex, BlockState.withPids, BlockState.withNumPids,
        GridIndex.toPids, grid, Grid.rank, h, hh]
  · intro r o; rfl
  · intro ax
    by_cases h : ax = 0
    · subst ax; rfl
    · have hh : ¬ax < 1 := by omega
      simp [programState, BlockState.withGridIndex, BlockState.withPids, BlockState.withNumPids,
        Grid.toNumPids, grid, Grid.rank, h, hh]

#axiomsClean program_run
#axiomsClean runPrograms_correct
#axiomsClean complete_schedule_correct
#axiomsClean full_grid_correct
#axiomsClean empty_grid
#axiomsClean source_no_wrap_bound
#axiomsClean programState_eq_grid
#specNonCircular Values avoiding [kernel, runPrograms]
#specNonCircular Frame avoiding [kernel, runPrograms]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DropoutTriton.RealGrid
