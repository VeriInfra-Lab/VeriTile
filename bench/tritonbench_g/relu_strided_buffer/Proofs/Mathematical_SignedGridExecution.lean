import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_SignedGrid
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_SignedTile

/-!
The actual FP32 load → explicit identity cast → where(x > 0, x, 0) →
explicit identity cast → typed store discharges every local execution premise
of SignedGrid. The resulting whole-grid theorem has NO scalar-output callback
or tile-result assumption. It executes the concrete tile body in a complete
ordering of the capped source tiles.

This is composition in the current mathematical FP32 evaluator. Native scalar
conformance, finite-width indexing, original branch lowering, and host metadata
are not inferred from this theorem. Empty-input host failure is handled outside
this tile executor, whose empty schedule itself is simply the identity.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SignedGridExecution
open VeriTile.Triton SignedView SignedGrid CappedGrid SignedTile

/-- Sequencing complete CTA tile lists has the same effect as the flattened
schedule. This connects the canonical pid-outer/step-inner source ordering. -/
theorem runTiles_append (exec : Nat → BlockState → Option BlockState)
    (first second : List Nat) (s : BlockState) :
    runTiles exec (first ++ second) s =
      (runTiles exec first s).bind (runTiles exec second) := by
  induction first generalizing s with
  | nil => rfl
  | cons start rest ih =>
    simp only [List.cons_append, runTiles]
    cases h : exec start s with
    | none => rfl
    | some middle => simpa using ih middle

theorem cappedStarts_one_tile {n B : Nat} (hone : tilesPerCTA n B = 1) :
    cappedStarts n B = (List.range (numCTAs n B)).map (fun pid => pid * B) := by
  simp [cappedStarts, hone]
  generalize List.range (numCTAs n B) = pids
  induction pids with
  | nil => rfl
  | cons pid pids ih => simpa using congrArg (List.cons (pid * B)) ih

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SignedGridExecution
