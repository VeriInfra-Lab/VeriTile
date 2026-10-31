import bench.tritonbench_g.sgmv_expand_slice.Sources.Mathematical_FPPrefix
namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

theorem fold_memory_congr {ι : Type} (l : List ι) (output : RegionName) (offset : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (v : ι → Cell Word) (s t : State Word) (h : s.mem=t.mem) :
    (l.foldl (fun u i => if active i then u.write output (offset i) (v i) else u) s).mem =
    (l.foldl (fun u i => if active i then u.write output (offset i) (v i) else u) t).mem := by
  induction l generalizing s t with
  | nil => exact h
  | cons i tail ih =>
    simp only [List.foldl_cons]
    split <;> apply ih
    · simp only [State.write,h]
    · exact h

end
end VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
