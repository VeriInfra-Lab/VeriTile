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

theorem fp32_tile_contract (input output : View) {n B capacity : Nat}
    (hout : output.Valid n capacity) (hstride : output.stride ≠ 0) :
    TileContract input output n B (fun value => storedCell32 (relu32 value))
      (fun start s => stepStmts (tileBody input output n start B) s) := by
  intro start s
  exact tile_success input output hout hstride s

/-- Complete signed-memory result from actual concrete FP32 tile execution.
Every active result is derived from this launch's initial input value. -/
theorem fp32_capped_tiles (input output : View) {n B inputCapacity outputCapacity : Nat}
    (hin : input.Valid n inputCapacity) (hout : output.Valid n outputCapacity)
    (hstride : output.stride ≠ 0) (hB : 0 < B)
    (hsep : FootprintsDisjoint input output n)
    (starts : List Nat) (hschedule : starts.Perm (cappedStarts n B)) (s : BlockState) :
    ∃ final, runTiles (fun start s => stepStmts (tileBody input output n start B) s)
        starts s = some final ∧
      (∀ i, i < n → final.mem output.region (output.address i) =
        storedCell32 (relu32 (s.readMemValue .fp32 input.region (input.address i)))) ∧
      (∀ i, i < n → final.mem input.region (input.address i) =
        s.mem input.region (input.address i)) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ output.address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < n → input.address i < inputCapacity ∧ output.address i < outputCapacity) := by
  exact capped_grid_success hin hout hstride hB hsep
    (fun value => storedCell32 (relu32 value))
    (fun start s => stepStmts (tileBody input output n start B) s)
    (fp32_tile_contract input output hout hstride) starts hschedule s

/-- Independent max(0,x) formula on represented finite input values. The input
premise binds the original read cells, not a supplied output or final state. -/
theorem fp32_capped_real_values (input output : View) {n B inputCapacity outputCapacity : Nat}
    (hin : input.Valid n inputCapacity) (hout : output.Valid n outputCapacity)
    (hstride : output.stride ≠ 0) (hB : 0 < B)
    (hsep : FootprintsDisjoint input output n)
    (starts : List Nat) (hschedule : starts.Perm (cappedStarts n B))
    (s : BlockState) (xs : Nat → ℝ)
    (hvalues : ∀ i, i < n → s.readMemValue .fp32 input.region (input.address i) = some (xs i)) :
    ∃ final, runTiles (fun start s => stepStmts (tileBody input output n start B) s)
        starts s = some final ∧
      (∀ i, i < n → final.mem output.region (output.address i) =
        storedCell32 (some (max 0 (xs i)))) ∧
      (∀ i, i < n → final.mem input.region (input.address i) =
        s.mem input.region (input.address i)) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ output.address i) →
        final.mem r a = s.mem r a) := by
  obtain ⟨final, hexec, hresult, hinput, hframe, _⟩ :=
    fp32_capped_tiles input output hin hout hstride hB hsep starts hschedule s
  refine ⟨final, hexec, ?_, hinput, hframe⟩
  intro i hi
  rw [hresult i hi, hvalues i hi, relu32_some]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SignedGridExecution
