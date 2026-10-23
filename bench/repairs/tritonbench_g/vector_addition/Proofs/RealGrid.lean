import bench.repairs.tritonbench_g.vector_addition.Kernels
import bench.tritonbench_g.vector_addition.Proofs.Real
import bench.tritonbench_g.vector_addition.Proofs.HostGrid
import VeriTile.Triton.Launch.Grid
import VeriTile.Meta.StatementAudit

/-! Whole mathematical launch of the original vector_addition kernel at BLOCK1024. Program steps
carry the actual memory left by earlier programs. Logical region equality is
allowed; this does not model shifted physical-byte aliases or interleavings. -/
namespace VeriTile.Bench.Repairs.VectorAddition.Contiguous.RealGrid
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.VectorAddition
noncomputable section
open Classical
set_option maxHeartbeats 4000000

def count (N : Nat) : Nat := (N + 1024 - 1) / 1024
def grid (N : Nat) : Grid := ⟨[count N]⟩
def kernel (X Y O : RegionName) (N : Nat) : Kernel :=
  (VeriTile.Bench.Repairs.VectorAddition.Contiguous.add_kernel X Y O N 1024).toAlgKernel

/-- Only launch metadata changes; memory is the current evolving memory. -/
def programState (s : BlockState) (N pid : Nat) : BlockState :=
  (s.withPids (fun ax => if ax = 0 then pid else 0)).withNumPids
    (fun ax => if ax = 0 then count N else 1)

def runPrograms (X Y O : RegionName) (N : Nat) : List Nat → BlockState → Option BlockState
  | [], s => some s
  | pid :: rest, s => do
      let t ← exec (kernel X Y O N) (programState s N pid)
      runPrograms X Y O N rest t

def Frame (O : RegionName) (N : Nat) (programs : List Nat) (s t : BlockState) : Prop :=
  ∀ r o, (r ≠ O ∨ N ≤ o ∨ o / 1024 ∉ programs) → t.mem r o = s.mem r o

def Values (X Y O : RegionName) (N : Nat) (programs : List Nat) (s t : BlockState) : Prop :=
  ∀ o, o < N → o / 1024 ∈ programs →
    t.readMem O o = s.readMem X o + s.readMem Y o

theorem read_eq_of_mem {s t : BlockState} {r : RegionName} {o : Nat}
    (h : t.mem r o = s.mem r o) : t.readMem r o = s.readMem r o := by
  simp only [BlockState.readMem, h]

/-- Reuses the established original DSL execution theorem, including its raw
frame. Both input tiles are loaded before the output scatter. -/
theorem program_run (X Y O : RegionName) (N pid : Nat) (s : BlockState) :
    ∃ t, exec (kernel X Y O N) (programState s N pid) = some t ∧
      Values X Y O N [pid] s t ∧ Frame O N [pid] s t := by
  obtain ⟨t, he, hv, hf⟩ := add_kernel_region_run X Y O N 1024 (programState s N pid)
    (fun j => s.readMem X (pid * 1024 + j.val))
    (fun j => s.readMem Y (pid * 1024 + j.val))
    (by intro j _; rfl) (by intro j _; rfl)
  refine ⟨t, he, ?_, ?_⟩
  · intro o ho hp
    have hp' : o / 1024 = pid := by simpa using hp
    let j : Fin 1024 := ⟨o % 1024, Nat.mod_lt _ (by decide)⟩
    have hj : pid * 1024 + j.val = o := by dsimp [j]; omega
    have hh := hv j (by simpa [programState, hj] using ho)
    simpa [programState, hj] using hh
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

/-- Distinct program tiles cannot change either input cell of a later tile.
This includes X=Y, X=O, Y=O and all three logical regions equal. -/
theorem runPrograms_correct (X Y O : RegionName) (N : Nat) (programs : List Nat)
    (unique : programs.Nodup) (s : BlockState) :
    ∃ t, runPrograms X Y O N programs s = some t ∧
      Values X Y O N programs s t ∧ Frame O N programs s t := by
  induction programs generalizing s with
  | nil => exact ⟨s, rfl, by simp [Values], by intro r o _; rfl⟩
  | cons pid rest ih =>
    have hp : pid ∉ rest := (List.nodup_cons.mp unique).1
    have hr : rest.Nodup := (List.nodup_cons.mp unique).2
    obtain ⟨mid, hm, hvm, hfm⟩ := program_run X Y O N pid s
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
        have hx := read_eq_of_mem (hfm X o (Or.inr (Or.inr hnot)))
        have hy := read_eq_of_mem (hfm Y o (Or.inr (Or.inr hnot)))
        rw [hvt o ho htail, hx, hy]
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
theorem complete_schedule_correct (X Y O : RegionName) (N : Nat) (programs : List Nat)
    (schedule : programs.Perm (List.range (count N))) (s : BlockState) :
    ∃ t, runPrograms X Y O N programs s = some t ∧
      (∀ o, o < N → t.readMem O o = s.readMem X o + s.readMem Y o) ∧
      (∀ r o, r ≠ O ∨ N ≤ o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hv, hf⟩ := runPrograms_correct X Y O N programs
    (schedule.nodup_iff.mpr List.nodup_range) s
  refine ⟨t, he, ?_, ?_⟩
  · intro o ho
    have cover := VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.grid_covers (⟨X, Y, O, N, 1024⟩ : VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.Parameters)
      (by change 0 < (1024 : Nat); decide)
    change N ≤ count N * 1024 at cover
    have owner : o / 1024 < count N := by omega
    exact hv o ho (schedule.mem_iff.mpr (List.mem_range.mpr owner))
  · intro r o h
    apply hf r o
    rcases h with h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)

theorem full_grid_correct (X Y O : RegionName) (N : Nat) (s : BlockState) :
    ∃ t, runPrograms X Y O N (List.range (count N)) s = some t ∧
      (∀ o, o < N → t.readMem O o = s.readMem X o + s.readMem Y o) ∧
      (∀ r o, r ≠ O ∨ N ≤ o → t.mem r o = s.mem r o) :=
  complete_schedule_correct X Y O N _ (List.Perm.refl _) s

theorem empty_grid (X Y O : RegionName) (s : BlockState) :
    runPrograms X Y O 0 (List.range (count 0)) s = some s := by
  rfl

#axiomsClean program_run
#axiomsClean runPrograms_correct
#axiomsClean complete_schedule_correct
#axiomsClean full_grid_correct
#axiomsClean empty_grid
#specNonCircular Values avoiding [kernel, runPrograms]
#specNonCircular Frame avoiding [kernel, runPrograms]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VectorAddition.Contiguous.RealGrid
