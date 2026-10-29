import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.TypedExecution
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 25000000
set_option maxRecDepth 48000
set_option linter.unusedSimpArgs false

def channelWrite {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (c : Fin 11) (t : State α) : State α :=
  (TileShape.allIndices [p.blockSize]).foldl (fun u i =>
    if outputOffset p s c i < p.outputNumel then
      u.write p.output (outputOffset p s c i) (.mk .real (expected M p s c i)) else u) t

def written {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  (List.finRange 11).foldl (fun t c => channelWrite M p s c t) s

private theorem fold_memory {α ι : Type} (l : List ι) (r : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).mem =
      l.foldl (fun mem i => fun r' o =>
        if active i ∧ r' = r ∧ o = off i then value i else mem r' o) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    by_cases hi : active i
    · simp only [List.foldl_cons, if_pos hi, ih]
      congr 1
      funext r' o
      simp [hi, State.write]
    · simp only [List.foldl_cons, if_neg hi, ih]
      congr 1
      funext r' o
      simp [hi]

/-- Memory of the full source equals the eleven independently specified
channel scatters, preserving all original operation order and masks. -/
theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    (execution M p s).val.mem = (written M p s).mem := by
  simp (config := {maxSteps := 4000000}) [execution, written, channelWrite,
    List.finRange_succ, List.foldl_map, fold_memory, expected, channels, loaded,
    coordOffset, outputOffset, bop, Region.cast]
  try rfl

#axiomsClean execution_memory
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP
