import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Reference
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

theorem maskedMemory_congr {α ι : Type} (l : List ι) (s t : State α)
    (R : RegionName) (offset : ι → Nat) (active : ι → Prop) [DecidablePred active]
    (value : ι → Cell α) (h : s.mem=t.mem) :
    (l.foldl (fun u i => if active i then u.write R (offset i) (value i) else u) s).mem =
    (l.foldl (fun u i => if active i then u.write R (offset i) (value i) else u) t).mem := by
  induction l generalizing s t with
  | nil => exact h
  | cons i tail ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i
    · simp only [hi,↓reduceIte]
      exact ih _ _ (by simp only [State.write,h])
    · simp only [hi,↓reduceIte]
      exact ih s t h

/-- Exact masked write order including any address aliasing. Division by zero,
NaNs and narrow-format conversion are left to the numerical primitives. -/
def outputMemory {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) (c : Config) (s : State α)
    (a : Accumulator α c.BM c.BD) : State α :=
  (TileShape.allIndices [c.BM,c.BD]).foldl (fun t i =>
    if rowStart c s+i.1.val < sequenceLength c s then
      t.write c.out (queryBase c s+(rowStart c s+i.1.val)*c.som+i.2.1.val*c.sok)
        (.mk .real (ofFloat .real (M.cast (some .fp32) c.storage.float .real (M.cast (some .fp32) .real c.storage.float
          (if D.equal (a.normalizer (i.1,())) (M.literal (some .fp32) .real 0) then M.literal (some .fp32) .real 0
           else M.binary (some .fp32) .real .div (a.output i) (a.normalizer (i.1,())))))))
    else t) s

#axiomsClean maskedMemory_congr
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source
