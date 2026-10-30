import bench.tritonbench_g.context_attn_bloom.Proofs.FPCore

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.FPSuccess
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
noncomputable section
variable {α : Type} [Inhabited α]

def scatter (p : Layout) (r : Regions) (initial s : State α)
    (v : Values α .real [p.BM,p.D]) : State α := by
  classical
  exact (TileShape.allIndices [p.BM,p.D]).foldl (fun t i =>
    if active p r initial i then t.write r.out (offset p r initial i) (.mk .real (v i)) else t) s

theorem masked_fold_mem_congr {I : Type} (rg : RegionName) (off : I → Nat)
    (v : I → Cell α) (on : I → Prop) [DecidablePred on] (lanes : List I)
    (s t : State α) (h : s.mem = t.mem) :
    (lanes.foldl (fun u i => if on i then u.write rg (off i) (v i) else u) s).mem =
    (lanes.foldl (fun u i => if on i then u.write rg (off i) (v i) else u) t).mem := by
  induction lanes generalizing s t with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : on i
    · simp only [hi,↓reduceIte]
      apply ih
      simp only [State.write]
      rw [h]
    · simp only [hi,↓reduceIte]
      exact ih s t h

theorem scatter_mem_congr (p : Layout) (r : Regions) (initial s t : State α)
    (v : Values α .real [p.BM,p.D]) (h : s.mem = t.mem) :
    (scatter p r initial s v).mem = (scatter p r initial t v).mem := by
  classical
  unfold scatter
  generalize TileShape.allIndices [p.BM,p.D] = lanes
  induction lanes generalizing s t with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active p r initial i
    · simp only [hi,↓reduceIte]
      apply ih
      simp only [State.write]
      rw [h]
    · simp only [hi,↓reduceIte]
      exact ih s t h

theorem scatter_frame (p : Layout) (r : Regions) (initial s : State α)
    (v : Values α .real [p.BM,p.D]) (rg : RegionName) (o : Nat)
    (hmiss : ∀ i, active p r initial i → rg = r.out → o = offset p r initial i → False) :
    (scatter p r initial s v).mem rg o = s.mem rg o := by
  classical
  apply State.masked_scatter_frame
  by_cases hr : rg = r.out
  · exact Or.inr (fun i _ hi ho => hmiss i hi hr ho)
  · exact Or.inl hr

end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.FPSuccess
