import bench.repairs.tritonbench_g.sin_computation.Proofs.Contract
import bench.tritonbench_g.add_example.Proofs.FiniteIndices

/-! Single-launch specialization of the existing finite source/index interface. -/
namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.Index
open VeriTile.Bench.TritonBenchG.AddExample.Finite

def ValidJob (n pid : Nat) : Prop := n < 2^31 ∧ pid < gridCount n

theorem source_abi (n pid : Nat) (h : ValidJob n pid) (lane : Nat) (hl : lane < 4) :
    n < 2^31 ∧ (offset32 pid lane).toInt = (pid*4+lane : Nat) ∧
      mask32 n pid lane = decide (pid*4+lane < n) :=
  ⟨h.1, no_wrap_indices n pid lane h.1 h.2 hl⟩

theorem active_iff (n pid : Nat) (h : ValidJob n pid) (lane : Nat) :
    lane ∈ programLanes n pid ↔ lane < 4 ∧ pid*4+lane < n := by
  simp only [programLanes, List.mem_filter, List.mem_range]
  constructor
  · rintro ⟨hl, hm⟩
    rw [(source_abi n pid h lane hl).2.2] at hm
    exact ⟨hl, of_decide_eq_true hm⟩
  · rintro ⟨hl, hi⟩
    exact ⟨hl, by rw [(source_abi n pid h lane hl).2.2]; exact decide_eq_true hi⟩

theorem active_bound (n pid : Nat) (h : ValidJob n pid) (lane : Nat)
    (hl : lane ∈ programLanes n pid) : pid*4+lane < n :=
  ((active_iff n pid h lane).mp hl).2

def indices (n pid : Nat) : List Nat := (programLanes n pid).map (fun lane => pid*4+lane)

theorem indices_bound (n pid : Nat) (h : ValidJob n pid) (i : Nat)
    (hi : i ∈ indices n pid) : i < n := by
  obtain ⟨lane, hl, rfl⟩ := List.mem_map.mp hi
  exact active_bound n pid h lane hl

structure Schedule (n : Nat) where
  bound : n < 2^31
  programs : List Nat
  complete : programs.Perm (List.range (gridCount n))

def jobs (n : Nat) (s : Schedule n) : List Nat := s.programs

theorem valid_of_mem (n : Nat) (s : Schedule n) (pid : Nat) (hp : pid ∈ jobs n s) :
    ValidJob n pid := ⟨s.bound, List.mem_range.mp (s.complete.mem_iff.mp hp)⟩

theorem covered (n : Nat) (s : Schedule n) (i : Nat) (hi : i < n) :
    ∃ pid ∈ jobs n s, i ∈ indices n pid := by
  obtain ⟨hp, hl, he⟩ := active_owner n i hi
  refine ⟨i/4, s.complete.mem_iff.mpr (List.mem_range.mpr hp), ?_⟩
  apply List.mem_map.mpr
  exact ⟨i%4, (active_iff n (i/4) ⟨s.bound,hp⟩ (i%4)).mpr ⟨hl, by omega⟩, he⟩

end VeriTile.Bench.Repairs.SinComputation.Contiguous.Index
