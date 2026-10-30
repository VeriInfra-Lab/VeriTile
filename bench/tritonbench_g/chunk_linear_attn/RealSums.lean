import bench.tritonbench_g.chunk_linear_attn.RealChunked
import bench.tritonbench_g.chunk_delta_fwd.Sources.RealModel
import Mathlib.Algebra.BigOperators.Intervals
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Real
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
open VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealMath (block_sum)

theorem sum_prefix (T n : Nat) (f : Nat → ℝ) :
    (∑ t : Fin n, if t.val<T then f t.val else 0)=
      ∑ t : Fin T, if t.val<n then f t.val else 0 := by
  rw [Fin.sum_univ_eq_sum_range (fun t => if t<T then f t else 0) n,
    Fin.sum_univ_eq_sum_range (fun t => if t<n then f t else 0) T,
    ← Finset.sum_filter,← Finset.sum_filter]
  congr 1
  ext t
  simp only [Finset.mem_filter,Finset.mem_range]
  omega

theorem blocks_prefix (T n width : Nat) (f : Nat → ℝ) :
    (∑ c : Fin n, ∑ t : Fin width, if c.val*width+t.val<T then f (c.val*width+t.val) else 0)=
      ∑ t : Fin T, if t.val<n*width then f t.val else 0 := by
  rw [block_sum (fun t => if t<T then f t else 0),sum_prefix]

theorem blocks_complete (T n width : Nat) (h : T≤n*width) (f : Nat → ℝ) :
    (∑ c : Fin n, ∑ t : Fin width, if c.val*width+t.val<T then f (c.val*width+t.val) else 0)=
      ∑ t : Fin T, f t.val := by
  rw [blocks_prefix]
  apply Finset.sum_congr rfl
  intro t _
  simp [show t.val<n*width from lt_of_lt_of_le t.isLt h]

theorem block_quotient (c width lane : Nat) (hl : lane<width) :
    (c*width+lane)/width=c := by
  have hp : 0<width := by omega
  simp [Nat.mul_comm c width,Nat.mul_add_div hp,Nat.div_eq_of_lt hl]

theorem blocks_select (T n width : Nat) (hp : 0<width) (h : T≤n*width)
    (select : Nat → Prop) [DecidablePred select] (f : Nat → ℝ) :
    (∑ c : Fin n, if select c.val then
      ∑ t : Fin width, if c.val*width+t.val<T then f (c.val*width+t.val) else 0 else 0)=
      ∑ t : Fin T, if select (t.val/width) then f t.val else 0 := by
  calc
    _ = ∑ c : Fin n, ∑ t : Fin width, if c.val*width+t.val<T then
        (if select ((c.val*width+t.val)/width) then f (c.val*width+t.val) else 0) else 0 := by
      apply Finset.sum_congr rfl
      intro c _
      by_cases hs : select c.val
      · simp only [if_pos hs]
        apply Finset.sum_congr rfl
        intro t _
        simp [block_quotient c.val width t.val t.isLt,hs]
      · simp only [if_neg hs]
        symm
        apply Finset.sum_eq_zero
        intro t _
        simp [block_quotient c.val width t.val t.isLt,hs]
    _ = _ := blocks_complete T n width h (fun t => if select (t/width) then f t else 0)

theorem reverse_prefix (N n : Nat) (hn : n≤N) (f : Nat → ℝ) :
    (∑ c ∈ Finset.range n, f (N-1-c)) =
      ∑ c ∈ Finset.range N, if N-n≤c then f c else 0 := by
  rw [← Finset.sum_range_reflect (fun c => if N-n≤c then f c else 0) N]
  have he : ∀ c∈Finset.range N, (if N-n≤N-1-c then f (N-1-c) else 0)=
      (if c<n then f (N-1-c) else 0) := by
    intro c hc
    have hc' : c<N := Finset.mem_range.mp hc
    have he : N-n≤N-1-c ↔ c<n := by omega
    simp only [he]
  rw [Finset.sum_congr rfl he,← Finset.sum_filter]
  have hs : (Finset.range N).filter (fun c => c<n)=Finset.range n := by
    ext c
    simp only [Finset.mem_filter,Finset.mem_range]
    omega
  rw [hs]

#axiomsClean blocks_prefix
#axiomsClean blocks_complete
#axiomsClean blocks_select
#axiomsClean reverse_prefix
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Real
