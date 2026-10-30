import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_SourceForRange
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_SourceSignedBranch

/-!
Both complete source constexpr branches, instantiated with the proved signed
dynamic perTile body. All source control/index assignments are retained, and
the grid-stride branch executes the actual forRangeDyn statement. The signed
perTile lowering uses the complete signed address before conversion to Nat.

The result is mem/undef observation equality to the concrete signed FP32 tile
executor. No output-value, complete-loop, or final-memory oracle is assumed.
This is not an equality of source scratch registers or native IEEE execution.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceGridExecution
open VeriTile.Triton SignedTile SignedGrid SourceForRange SourceSignedBranch

theorem signed_tile_implementation (c : SignedViews) :
    TileImplementation c.inputView c.outputView c.n c.B c.perTile := by
  intro T s htile
  have hcontrol := sourceFinal_control c T s
  exact ⟨c.sourceFinal T s, sourceTile_exec c T s htile,
    sourceFinal_obs c T s, hcontrol.1, hcontrol.2.1⟩

theorem signed_tile_observations (c : SignedViews) (start : Nat)
    (s t : BlockState) (h : ObsEq s t) :
    ObsEq (finalState c.inputView c.outputView c.n start c.B s)
      (finalState c.inputView c.outputView c.n start c.B t) :=
  SourceBranch.tile_obs_congr c.inputView c.outputView c.n start c.B h

def kernelBody (c : SignedViews) (oneTile : Bool) (tiles : Nat) : List Stmt :=
  if oneTile then oneTileBody c.perTile c.n c.B else gridBody c.perTile c.n c.B tiles

noncomputable def executeKernel (c : SignedViews) (oneTile : Bool) (tiles : Nat)
    (s : BlockState) : Option BlockState := stepStmts (kernelBody c oneTile tiles) s

def kernelStarts (c : SignedViews) (oneTile : Bool) (tiles : Nat) (s : BlockState) : List Nat :=
  if oneTile then [s.pids 0 * c.B]
  else (List.range tiles).map (fun j => (s.pids 0 + j * s.numPids 0) * c.B)

/-- A single CTA executes the original source control flow for either branch.
Its observed memory/padding equals the corresponding concrete tile sequence.
No local implementation or congruence premise remains in this theorem. -/
theorem kernel_execution (c : SignedViews) (oneTile : Bool) (tiles : Nat) (s : BlockState) :
    ∃ actualFinal flatFinal,
      executeKernel c oneTile tiles s = some actualFinal ∧
      runTiles (fun start state => stepStmts
        (tileBody c.inputView c.outputView c.n start c.B) state)
        (kernelStarts c oneTile tiles s) s = some flatFinal ∧
      ObsEq actualFinal flatFinal := by
  cases oneTile
  · exact grid_branch_execution c.inputView c.outputView c.n c.B c.perTile
      (signed_tile_implementation c) (signed_tile_observations c) tiles s
  · obtain ⟨actualFinal, hexec, hobs⟩ :=
      one_tile_execution c.inputView c.outputView c.n c.B c.perTile
        (signed_tile_implementation c) (signed_tile_observations c) s
    refine ⟨actualFinal, finalState c.inputView c.outputView c.n (s.pids 0 * c.B) c.B s,
      hexec, ?_, hobs⟩
    simp only [kernelStarts, if_true, runTiles, tile_exec, Option.bind_some]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceGridExecution
