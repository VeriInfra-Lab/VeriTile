import bench.tritonbench_g.chunk_linear_attn.RealSums
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Real
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

theorem one_block (T N width c : Nat) (hp : 0<width) (hcover : T≤N*width) (hc : c<N)
    (f : Nat → ℝ) :
    (∑ lane : Fin width, if c*width+lane.val<T then f (c*width+lane.val) else 0)=
      ∑ t : Fin T, if t.val/width=c then f t.val else 0 := by
  have h := blocks_select T N width hp hcover (fun b => b=c) f
  have he : ∀ b : Fin N, b.val=c ↔ b=⟨c,hc⟩ := by
    intro b
    constructor
    · intro he; exact Fin.ext he
    · intro he; exact congrArg Fin.val he
  simpa only [he,Finset.sum_ite_eq',Finset.mem_univ,ite_true] using h

theorem causal_block (T N width c row : Nat) (hp : 0<width) (hcover : T≤N*width) (hc : c<N)
    (f : Nat → ℝ) :
    (∑ lane : Fin width, if lane.val≤row then
      (if c*width+lane.val<T then f (c*width+lane.val) else 0) else 0)=
      ∑ t : Fin T, if t.val/width=c ∧ t.val≤c*width+row then f t.val else 0 := by
  calc
    _ = ∑ lane : Fin width, if c*width+lane.val<T then
        (if c*width+lane.val≤c*width+row then f (c*width+lane.val) else 0) else 0 := by
      apply Finset.sum_congr rfl
      intro lane _
      by_cases ht : c*width+lane.val<T <;> by_cases hl : lane.val≤row <;> simp [ht,hl]
    _ = ∑ t : Fin T, if t.val/width=c then
        (if t.val≤c*width+row then f t.val else 0) else 0 := one_block T N width c hp hcover hc (fun t => if t≤c*width+row then f t else 0)
    _ = _ := by
      apply Finset.sum_congr rfl
      intro t _
      by_cases ht : t.val/width=c <;> by_cases hl : t.val≤c*width+row <;> simp [ht,hl]

theorem future_block (T N width c row : Nat) (hp : 0<width) (hcover : T≤N*width) (hc : c<N)
    (f : Nat → ℝ) :
    (∑ lane : Fin width, if row≤lane.val then
      (if c*width+lane.val<T then f (c*width+lane.val) else 0) else 0)=
      ∑ t : Fin T, if t.val/width=c ∧ c*width+row≤t.val then f t.val else 0 := by
  calc
    _ = ∑ lane : Fin width, if c*width+lane.val<T then
        (if c*width+row≤c*width+lane.val then f (c*width+lane.val) else 0) else 0 := by
      apply Finset.sum_congr rfl
      intro lane _
      by_cases ht : c*width+lane.val<T <;> by_cases hl : row≤lane.val <;> simp [ht,hl]
    _ = ∑ t : Fin T, if t.val/width=c then
        (if c*width+row≤t.val then f t.val else 0) else 0 := one_block T N width c hp hcover hc (fun t => if c*width+row≤t then f t else 0)
    _ = _ := by
      apply Finset.sum_congr rfl
      intro t _
      by_cases ht : t.val/width=c <;> by_cases hl : c*width+row≤t.val <;> simp [ht,hl]

#axiomsClean one_block
#axiomsClean causal_block
#axiomsClean future_block
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Real
