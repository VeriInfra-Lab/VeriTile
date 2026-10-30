import bench.tritonbench_g.quantize_global.Proofs.Model
import bench.tritonbench_g.quantize_global.Proofs.Mathematical_SourceIndices
namespace QuantizeGlobalRepair
set_option maxHeartbeats 1000000
set_option maxRecDepth 24000

theorem segment_cover (n i : Nat) (hi : i < n) :
    i / span < segmentCount n ∧ i % span < segmentSize n (i / span) := by
  change i/1073741824 < (n+1073741824-1)/1073741824 ∧
    i%1073741824 < min (n-i/1073741824*1073741824) 1073741824
  omega

theorem local_cover (n k i : Nat) (c : Config) (hi : i < segmentSize n k) :
    i ∈ QuantizeGlobalFinite.footprint c (List.range (programs n k c)) := by
  rw [QuantizeGlobalFinite.footprint_membership, List.mem_range]
  simp only [programs, QuantizeGlobalFinite.programs]
  generalize segmentSize n k = size at *
  cases c <;> simp only [QuantizeGlobalFinite.Config.block]
  all_goals omega

theorem schedule_covers (n : Nat) (trials : Nat → List Config)
    (nonempty : ∀ k, k < segmentCount n → trials k ≠ []) (i : Nat) (hi : i < n) :
    i ∈ footprint (schedule n trials) := by
  obtain ⟨hk, hl⟩ := segment_cover n i hi
  obtain ⟨c, hc⟩ := List.exists_mem_of_ne_nil (trials (i/span)) (nonempty (i/span) hk)
  have hlocal := local_cover n (i/span) (i%span) c hl
  simp only [QuantizeGlobalFinite.footprint, List.mem_flatMap, List.mem_map,
    List.mem_range] at hlocal
  obtain ⟨pid, hp, lane, hlane, hoff⟩ := hlocal
  apply List.mem_flatMap.mpr
  refine ⟨⟨i/span*span,c,pid⟩, ?_, ?_⟩
  · apply List.mem_flatMap.mpr
    refine ⟨i/span, List.mem_range.mpr hk, ?_⟩
    apply List.mem_flatMap.mpr
    exact ⟨c,hc,List.mem_map.mpr ⟨pid,List.mem_range.mpr hp,rfl⟩⟩
  · apply List.mem_map.mpr
    refine ⟨lane,List.mem_range.mpr hlane,?_⟩
    dsimp only [Invocation.offset]
    simp only [QuantizeGlobalFinite.offset] at hoff
    have hdiv := Nat.mod_add_div i span
    rw [Nat.mul_comm span (i/span)] at hdiv
    omega

theorem grid_bound (n k : Nat) (c : Config) : programs n k c ≤ 1048576 := by
  have h := Nat.min_le_right (n-k*span) span
  cases c <;> simp only [programs, QuantizeGlobalFinite.programs, QuantizeGlobalFinite.Config.block,
    segmentSize,span] at * <;> omega

/-- Every evaluated lane, including the masked tail, fits signed64. This
proves every add/multiply operand and result nonnegative and bounded too. -/
theorem offsets_i64 (n k pid lane : Nat) (c : Config) (hn : n < 2^63)
    (hk : k < segmentCount n) (hp : pid < programs n k c) (hl : lane < c.block) :
    k*span + pid*c.block + lane < 2^63 := by
  have hsize := Nat.min_le_right (n-k*span) span
  norm_num only [Nat.reducePow] at hn ⊢
  cases c <;> simp only [programs, segmentCount, segmentSize, span,
    QuantizeGlobalFinite.programs, QuantizeGlobalFinite.Config.block] at * <;> omega

theorem safe_pointer (n offset : Nat) (hn : 0 < n) :
    (if offset < n then offset else 0) < n := by split <;> omega

theorem empty_schedule (trials : Nat → List Config) : schedule 0 trials = [] := by
  simp [schedule,segmentCount,span]

#axiomsClean schedule_covers
#axiomsClean grid_bound
#axiomsClean offsets_i64
#axiomsClean safe_pointer
end QuantizeGlobalRepair
