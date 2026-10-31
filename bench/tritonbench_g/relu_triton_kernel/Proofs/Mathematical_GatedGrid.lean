import bench.tritonbench_g.relu_triton_kernel.Proofs.Mathematical_FP
import VeriTile.Meta.StatementAudit

/-! Ordered launches of the original pid-zero-only store. Every program
executes against the preceding state, including its loads and comparisons.
No native instruction-interleaving or byte-address refinement is asserted. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.GatedGrid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

abbrev Memory (α : Type) := RegionName → Nat → Cell α

def atPid {α : Type} (s : State α) (pid : Nat) : State α :=
  { s with pids := fun ax => if ax = 0 then pid else s.pids ax }

def runPrefix {α : Type} [Inhabited α] (M : Algebra α)
    (kernel : ComputeKernel) (initial : State α) : Nat → Option (State α)
  | 0 => some initial
  | k + 1 => do
      let s ← runPrefix M kernel initial k
      FP.Structural.exec M kernel (atPid s k)

def gridCount (N B : Nat) : Nat := (N + B - 1) / B

def runGrid {α : Type} [Inhabited α] (M : Algebra α)
    (kernel : ComputeKernel) (N B : Nat) (initial : State α) : Option (State α) :=
  runPrefix M kernel initial (gridCount N B)

/-- Generic composition lemma, instantiated below by the actual source
execution theorem. Its reference has no access to an interpreter result. -/
theorem prefix_of_first_only {α : Type} [Inhabited α] (M : Algebra α)
    (kernel : ComputeKernel) (reference : Memory α → Memory α)
    (step_correct : ∀ (s : State α) (pid : Nat),
      ∃ t, FP.Structural.exec M kernel (atPid s pid) = some t ∧
        t.mem = if pid = 0 then reference s.mem else s.mem)
    (initial : State α) (count : Nat) :
    ∃ final, runPrefix M kernel initial count = some final ∧
      final.mem = if count = 0 then initial.mem else reference initial.mem := by
  induction count with
  | zero => exact ⟨initial, rfl, rfl⟩
  | succ k ih =>
    obtain ⟨s, hs, hm⟩ := ih
    obtain ⟨t, ht, htm⟩ := step_correct s k
    refine ⟨t, ?_, ?_⟩
    · simp only [runPrefix, hs, Option.bind_some]
      exact ht
    · cases k with
      | zero => simpa only [Nat.zero_eq, ↓reduceIte, hm] using htm
      | succ k => simpa only [Nat.succ_ne_zero, ↓reduceIte, hm] using htm

def legacyValue {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (x : RegionName) (mem : Memory α) (o : Nat) : Cell α :=
  .mk .real (let v := (mem x o).read .real
    if cmp (M.literal none .real 0) v then v else M.literal none .real 0)

/-- Exactly the cells written by pid zero; the allocated tail is arbitrary. -/
def legacyReference {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (x y : RegionName) (N B : Nat)
    (mem : Memory α) : Memory α :=
  fun r o => if r = y ∧ o < B ∧ o < N then legacyValue M cmp x mem o else mem r o

theorem legacy_reference_frame {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (x y : RegionName) (N B : Nat)
    (mem : Memory α) (r : RegionName) (o : Nat)
    (outside : r ≠ y ∨ B ≤ o ∨ N ≤ o) :
    legacyReference M cmp x y N B mem r o = mem r o := by
  simp only [legacyReference]
  rw [if_neg]
  intro h
  rcases outside with hr | hb | hn
  · exact hr h.1
  · omega
  · omega

theorem empty_grid_count (B : Nat) : gridCount 0 B = 0 := by
  unfold gridCount
  cases B with
  | zero => simp
  | succ B => exact Nat.div_eq_of_lt (by omega)

theorem positive_grid_count (N B : Nat) (hn : 0 < N) (hb : 0 < B) :
    0 < gridCount N B := by
  unfold gridCount
  exact Nat.div_pos (by omega) hb

#specNonCircular legacyReference avoiding [ runPrefix, runGrid, FP.Structural.exec]
#axiomsClean prefix_of_first_only
#axiomsClean legacy_reference_frame
#axiomsClean empty_grid_count
#axiomsClean positive_grid_count
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.GatedGrid
