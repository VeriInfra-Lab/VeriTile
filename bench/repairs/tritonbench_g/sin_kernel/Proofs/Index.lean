import bench.repairs.tritonbench_g.sin_kernel.Proofs.Contract
import bench.tritonbench_g.sin_kernel.Proofs.NativeSource

namespace VeriTile.Bench.Repairs.SinKernel
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite

theorem native_indices (x : TensorMetadata) (h : InputContract x)
    (pid lane : Nat) (hp : pid < gridCount x.numel) (hl : lane < 1024) :
    (SinKernelNative.sourceOffset x.numel pid lane).toInt = (pid*1024+lane : Nat) ∧
    SinKernelNative.mask x.numel pid lane = decide (pid*1024+lane < x.numel) := by
  rw [SinKernelNative.source_offset]
  exact (source_index_arithmetic x.numel pid lane h.extent hp hl).2

#axiomsClean native_indices
end VeriTile.Bench.Repairs.SinKernel

/-! Single-launch specialization of the existing finite source/index interface. -/
namespace VeriTile.Bench.Repairs.SinKernel.Index
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite
abbrev programLanes := SinKernelNative.activeTile

def ValidJob (n pid : Nat) : Prop := n < 2^31 ∧ pid < gridCount n

theorem source_abi (n pid : Nat) (h : ValidJob n pid) (lane : Nat) (hl : lane < 1024) :
    n < 2^31 ∧ (offset32 pid lane).toInt = (pid*1024+lane : Nat) ∧
      VeriTile.Bench.TritonBenchG.VectorAddition.Native.mask n pid lane = decide (pid*1024+lane < n) :=
  ⟨h.1, (source_index_arithmetic n pid lane h.1 h.2 hl).2⟩

theorem active_iff (n pid : Nat) (h : ValidJob n pid) (lane : Nat) :
    lane ∈ programLanes n pid ↔ lane < 1024 ∧ pid*1024+lane < n := by
  simp only [programLanes, SinKernelNative.activeTile,
    VeriTile.Bench.TritonBenchG.VectorAddition.Native.activeTile, List.mem_filter, List.mem_range]
  constructor
  · rintro ⟨hl, hm⟩
    rw [(source_abi n pid h lane hl).2.2] at hm
    exact ⟨hl, of_decide_eq_true hm⟩
  · rintro ⟨hl, hi⟩
    exact ⟨hl, by rw [(source_abi n pid h lane hl).2.2]; exact decide_eq_true hi⟩

theorem active_bound (n pid : Nat) (h : ValidJob n pid) (lane : Nat)
    (hl : lane ∈ programLanes n pid) : pid*1024+lane < n :=
  ((active_iff n pid h lane).mp hl).2

def indices (n pid : Nat) : List Nat := (programLanes n pid).map (fun lane => pid*1024+lane)

theorem indices_bound (n pid : Nat) (h : ValidJob n pid) (i : Nat)
    (hi : i ∈ indices n pid) : i < n := by
  obtain ⟨lane, hl, rfl⟩ := List.mem_map.mp hi
  exact active_bound n pid h lane hl

structure Schedule (n : Nat) where
  bound : n < 2^31
  programs : List Nat
  complete : programs.Perm (List.range (gridCount n))

def ascendingSchedule (x : TensorMetadata) (h : InputContract x) : Schedule x.numel :=
  ⟨h.extent, List.range (gridCount x.numel), List.Perm.refl _⟩

def jobs (n : Nat) (s : Schedule n) : List Nat := s.programs

theorem valid_of_mem (n : Nat) (s : Schedule n) (pid : Nat) (hp : pid ∈ jobs n s) :
    ValidJob n pid := ⟨s.bound, List.mem_range.mp (s.complete.mem_iff.mp hp)⟩

theorem covered (n : Nat) (s : Schedule n) (i : Nat) (hi : i < n) :
    ∃ pid ∈ jobs n s, i ∈ indices n pid := by
  obtain ⟨hp, hl, he⟩ := active_owner n i hi
  refine ⟨i/1024, s.complete.mem_iff.mpr (List.mem_range.mpr hp), ?_⟩
  apply List.mem_map.mpr
  exact ⟨i%1024, (active_iff n (i/1024) ⟨s.bound,hp⟩ (i%1024)).mpr ⟨hl, by omega⟩, he⟩

end VeriTile.Bench.Repairs.SinKernel.Index
