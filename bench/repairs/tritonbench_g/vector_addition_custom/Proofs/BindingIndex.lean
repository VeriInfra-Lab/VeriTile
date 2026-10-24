import bench.repairs.tritonbench_g.vector_addition_custom.Proofs.Contract
import bench.repairs.tritonbench_g.vector_addition_custom.Proofs.Source

/-! Single-launch specialization of the existing finite source/index interface. -/
namespace VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.Index
open Source

def ValidJob (n pid : Nat) : Prop := n < 2^31 ∧ pid < gridCount n

theorem source_abi (n pid : Nat) (h : ValidJob n pid) (lane : Nat) (hl : lane < 16) :
    n < 2^31 ∧ (offset32 pid lane).toInt = (pid*16+lane : Nat) ∧
      mask32 n pid lane = decide (pid*16+lane < n) :=
  ⟨h.1, VeriTile.Bench.TritonBenchG.VectorAdditionCustom.Native.bounded_arithmetic n pid lane h.1 h.2 hl⟩

theorem active_iff (n pid : Nat) (h : ValidJob n pid) (lane : Nat) :
    lane ∈ programLanes n pid ↔ lane < 16 ∧ pid*16+lane < n := by
  simp only [programLanes, List.mem_filter, List.mem_range]
  constructor
  · rintro ⟨hl, hm⟩
    rw [(source_abi n pid h lane hl).2.2] at hm
    exact ⟨hl, of_decide_eq_true hm⟩
  · rintro ⟨hl, hi⟩
    exact ⟨hl, by rw [(source_abi n pid h lane hl).2.2]; exact decide_eq_true hi⟩

theorem active_bound (n pid : Nat) (h : ValidJob n pid) (lane : Nat)
    (hl : lane ∈ programLanes n pid) : pid*16+lane < n :=
  ((active_iff n pid h lane).mp hl).2

def indices (n pid : Nat) : List Nat := (programLanes n pid).map (fun lane => pid*16+lane)

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
  have hp : i/16 < gridCount n := by unfold gridCount; omega
  have hl : i%16 < 16 := Nat.mod_lt _ (by decide)
  have he : i/16*16+i%16=i := by omega
  refine ⟨i/16, s.complete.mem_iff.mpr (List.mem_range.mpr hp), ?_⟩
  apply List.mem_map.mpr
  exact ⟨i%16, (active_iff n (i/16) ⟨s.bound,hp⟩ (i%16)).mpr ⟨hl, by omega⟩, he⟩


/-- Every admitted extent has a complete schedule; the public arbitrary-order
schedule parameter adds no tensor-domain restriction. -/
def canonicalSchedule (n : Nat) (bound : n < 2^31) : Schedule n :=
  ⟨bound, List.range (gridCount n), List.Perm.refl _⟩

/-- Distinct active source lanes write disjoint FP32 bytes. -/
theorem distinct_store_bytes (base p q i j a b : Nat)
    (hi : i < 16) (hj : j < 16) (ha : a < 4) (hb : b < 4)
    (different : p ≠ q ∨ i ≠ j) :
    base + 4*(p*16+i) + a ≠ base + 4*(q*16+j) + b := by
  intro same
  have indices : p*16+i = q*16+j := by omega
  have equal : p=q ∧ i=j := by omega
  rcases different with hp | hl
  · exact hp equal.1
  · exact hl equal.2

#axiomsClean distinct_store_bytes
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.Index
