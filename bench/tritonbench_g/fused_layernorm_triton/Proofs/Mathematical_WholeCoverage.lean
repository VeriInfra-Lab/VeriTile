import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeReal
import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeGeometry
import Mathlib.Data.List.Perm.Basic

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
open VeriTile.Triton FP.Structural FPStages
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem finRange_values (B : Nat) : (List.finRange B).map Fin.val = List.range B := by
  induction B with
  | zero => simp
  | succ B ih =>
    rw [List.finRange_succ, List.map_cons, List.map_map, List.range_succ_eq_map]
    rw [← ih, List.map_map]
    rfl

def emitted {α : Type} (D : Nat) (value : Nat → α) (j : Nat) : List α :=
  if j < D then [value j] else []

def chunkSamples (s : State ℝ) (X : RegionName) (D B k : Nat) : List ℝ :=
  (List.finRange B).flatMap fun i =>
    emitted D (fun j => (s.mem X (j + D * s.pids 0)).read .real) (k * B + i.val)

def collectedSamples (s : State ℝ) (X : RegionName) (D B : Nat) : Nat → Nat → List ℝ
  | _, 0 => []
  | start, count + 1 => chunkSamples s X D B start ++ collectedSamples s X D B (start + 1) count

theorem advance_samples_perm (s : State ℝ) (X : RegionName) (D B k : Nat)
    (previous : Fin B → List ℝ) :
    ((List.finRange B).flatMap (advanceSamples s X D B k previous)).Perm
      ((List.finRange B).flatMap previous ++ chunkSamples s X D B k) := by
  have he : advanceSamples s X D B k previous = fun i => previous i ++
      emitted D (fun j => (s.mem X (j + D * s.pids 0)).read .real) (k * B + i.val) := by
    funext i
    simp only [advanceSamples, emitted]
    split_ifs <;> simp
  rw [he]
  exact (List.flatMap_append_perm (List.finRange B) previous _).symm

theorem recurrence_samples_perm (s : State ℝ) (X : RegionName) (D B start count : Nat)
    (previous : Fin B → List ℝ) :
    ((List.finRange B).flatMap (sampleRecurrence s X D B start count previous)).Perm
      ((List.finRange B).flatMap previous ++ collectedSamples s X D B start count) := by
  induction count generalizing start previous with
  | zero => simp [sampleRecurrence, collectedSamples]
  | succ count ih =>
    have h1 := ih (start + 1) (advanceSamples s X D B start previous)
    have h2 := (advance_samples_perm s X D B start previous).append_right
      (collectedSamples s X D B (start + 1) count)
    simpa [sampleRecurrence, collectedSamples, List.append_assoc] using h1.trans h2

theorem chunk_samples_range (s : State ℝ) (X : RegionName) (D B k : Nat) :
    chunkSamples s X D B k = (List.range' (k * B) B).flatMap
      (emitted D (fun j => (s.mem X (j + D * s.pids 0)).read .real)) := by
  unfold chunkSamples
  rw [List.range'_eq_map_range, List.flatMap_map]
  calc
    _ = ((List.finRange B).map Fin.val).flatMap
        (fun j => emitted D (fun a => (s.mem X (a + D * s.pids 0)).read .real) (k * B + j)) := by
          rw [List.flatMap_map]
    _ = _ := by rw [finRange_values]

theorem collected_samples_range (s : State ℝ) (X : RegionName) (D B start count : Nat) :
    collectedSamples s X D B start count = (List.range' (start * B) (count * B)).flatMap
      (emitted D (fun j => (s.mem X (j + D * s.pids 0)).read .real)) := by
  induction count generalizing start with
  | zero => simp [collectedSamples]
  | succ count ih =>
    rw [collectedSamples, chunk_samples_range, ih, ← List.flatMap_append]
    congr 1
    rw [Nat.add_mul, one_mul, List.range'_append_1]
    congr 1
    simp [Nat.add_mul, Nat.add_comm]

theorem emitted_range {α : Type} (value : Nat → α) (D n : Nat) :
    (List.range n).flatMap (emitted D value) = (List.range (min D n)).map value := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.flatMap_append, ih]
    by_cases hn : n < D
    · have h1 : min D n = n := Nat.min_eq_right (by omega)
      have h2 : min D (n + 1) = n + 1 := Nat.min_eq_right (by omega)
      simp [emitted, hn, h1, h2, List.range_succ]
    · have h1 : min D n = D := Nat.min_eq_left (by omega)
      have h2 : min D (n + 1) = D := Nat.min_eq_left (by omega)
      simp [emitted, hn, h1, h2]

def allSamples (s : State ℝ) (X : RegionName) (D B : Nat) : Fin B → List ℝ :=
  sampleRecurrence s X D B 1 (blockCount D B - 1) (firstSamples s X D B)

theorem all_samples_perm (s : State ℝ) (X : RegionName) (D B : Nat)
    (hD : 0 < D) (hB : 0 < B) :
    ((List.finRange B).flatMap (allSamples s X D B)).Perm
      ((List.range D).map (fun j => (s.mem X (j + D * s.pids 0)).read .real)) := by
  have hp := recurrence_samples_perm s X D B 1 (blockCount D B - 1) (firstSamples s X D B)
  have hf : (List.finRange B).flatMap (firstSamples s X D B) = chunkSamples s X D B 0 := by
    unfold chunkSamples
    congr 1
    funext i
    simp [firstSamples, emitted]
  have hc : (blockCount D B - 1).succ = blockCount D B := by
    have := block_count_positive D B hD hB
    omega
  rw [hf] at hp
  have he : chunkSamples s X D B 0 ++ collectedSamples s X D B 1 (blockCount D B - 1) =
      (List.range D).map (fun j => (s.mem X (j + D * s.pids 0)).read .real) := by
    rw [← collectedSamples, hc, collected_samples_range]
    simpa [List.range_eq_range', Nat.min_eq_left (block_count_covers D B hB)] using
      emitted_range (fun j => (s.mem X (j + D * s.pids 0)).read .real) D (blockCount D B * B)
  simpa [allSamples, he] using hp

theorem moments_perm (v : WelfordTriple ℝ) (xs ys : List ℝ)
    (h : Moments v xs) (hp : xs.Perm ys) : Moments v ys := by
  constructor
  · exact h.weight.trans (congrArg Nat.cast hp.length_eq)
  · exact h.sum.trans hp.sum_eq
  · exact h.squares.trans (hp.map (fun x => x ^ 2)).sum_eq
  · intro he
    apply h.empty
    have hl := hp.length_eq
    rw [he] at hl
    simpa using hl

theorem whole_statistics_real_moments (rest : Algebra ℝ) (s : State ℝ) (X : RegionName)
    (D B : Nat) (hD : 0 < D) (hB : 0 < B) (tree : ReductionTree B) (ht : tree.Complete) :
    Moments (treeValue (ideal rest) realEqual (allLaneValues (ideal rest) s X D B) tree)
      ((List.range D).map (fun j => (s.mem X (j + D * s.pids 0)).read .real)) := by
  have hm := recurrence_moments rest s X D B 1 (blockCount D B - 1) _ _
    (first_moments rest s X D B)
  have hh := moments_tree _ _ hm tree
  rw [← ideal_tree rest _ tree] at hh
  have hp : (treeSamples (allSamples s X D B) tree).Perm
      ((List.range D).map (fun j => (s.mem X (j + D * s.pids 0)).read .real)) := by
    rw [tree_samples_flatMap]
    exact (ht.flatMap_right (allSamples s X D B)).trans (all_samples_perm s X D B hD hB)
  exact moments_perm _ _ _ hh hp

#axiomsClean all_samples_perm
#axiomsClean whole_statistics_real_moments
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
