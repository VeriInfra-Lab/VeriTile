import bench.tritonbench_g.fp4_to_bf16.Proofs.FP4Bits
import VeriTile.Triton.Core.Reshape

namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

private theorem flatMap_length {α β : Type} (l : List α) (f : α → List β)
    (k : Nat) (h : ∀ a ∈ l, (f a).length = k) : (l.flatMap f).length = l.length*k := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.flatMap_cons, List.length_append, List.length_cons]
    rw [h a (by simp), ih (by intro a ha; exact h a (by simp [ha]))]
    simp [Nat.add_mul, Nat.add_comm]

private theorem flatMap_get {α β : Type} (l : List α) (f : α → List β)
    (k : Nat) (h : ∀ a ∈ l, (f a).length = k)
    (q r : Nat) (hq : q < l.length) (hr : r < k) :
    (l.flatMap f)[q*k+r]'(by
      rw [flatMap_length l f k h]
      exact lt_of_lt_of_le (Nat.add_lt_add_left hr _) (by
        simpa [Nat.add_mul] using Nat.mul_le_mul_right k hq)) =
      (f l[q])[r]'(by rw [h l[q] (List.getElem_mem _)]; exact hr) := by
  induction l generalizing q with
  | nil => simp at hq
  | cons a l ih =>
    have ha := h a (by simp)
    have hl : ∀ b ∈ l, (f b).length = k := by intro b hb; exact h b (by simp [hb])
    cases q with
    | zero =>
      simp only [Nat.zero_mul, Nat.zero_add, List.getElem_cons_zero, List.flatMap_cons]
      exact List.getElem_append_left (by omega)
    | succ q =>
      have hq' : q < l.length := by simpa using hq
      simp only [List.getElem_cons_succ, List.flatMap_cons]
      rw [List.getElem_append_right (by simp only [Nat.add_mul, Nat.one_mul]; omega)]
      have he : (q+1)*k+r-(f a).length = q*k+r := by simp only [Nat.add_mul, Nat.one_mul]; omega
      simpa only [he] using ih hl q hq'

/-- Flattening join preserves consecutive high/low lanes for every block size. -/
theorem reshape_rows32 (block : Nat) (j : TileIndex [block*32])
    (h : TileShape.product [block,32] = TileShape.product [block*32]) :
    TileShape.reshapeIndex h j =
      (⟨j.1.val / 32, by have := j.1.isLt; omega⟩,
       ⟨j.1.val % 32, Nat.mod_lt _ (by decide)⟩, PUnit.unit) := by
  unfold TileShape.reshapeIndex
  simp only [TileShape.linearIndex, TileShape.product, Nat.mul_one, Nat.add_zero]
  let f := fun i : Fin block => (TileShape.allIndices [32]).map (fun is => (i,is))
  have hf : ∀ a ∈ List.finRange block, (f a).length = 32 := by
    intro a ha; simp only [f, List.length_map]; rfl
  have hq : j.1.val / 32 < (List.finRange block).length := by
    simp only [List.length_finRange]; have := j.1.isLt; omega
  have he : j.1.val / 32 * 32 + j.1.val % 32 = j.1.val := by omega
  have hg := flatMap_get (List.finRange block) f 32 hf (j.1.val/32) (j.1.val%32)
    hq (Nat.mod_lt _ (by decide))
  simp only [he] at hg
  change ((List.finRange block).flatMap f)[j.1.val]'(_ : j.1.val < _) = _
  refine hg.trans ?_
  simp only [f, List.getElem_finRange, List.getElem_map]
  simp [TileShape.allIndices, ← List.map_eq_flatMap, List.getElem_finRange]

/-- Reshaping a vector into rows selects its ordinary row-major index. -/
theorem reshape_vector (n : Nat) (shape : TileShape) (j : TileIndex shape)
    (h : TileShape.product [n] = TileShape.product shape) :
    TileShape.reshapeIndex h j =
      (⟨TileShape.linearIndex shape j, by
        have hn : n = TileShape.product shape := by simpa only [TileShape.product, Nat.mul_one] using h
        rw [hn]; exact TileShape.linearIndex_lt shape j⟩, PUnit.unit) := by
  unfold TileShape.reshapeIndex
  simp [TileShape.allIndices, ← List.map_eq_flatMap]

#axiomsClean reshape_vector

#axiomsClean reshape_rows32
end VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
