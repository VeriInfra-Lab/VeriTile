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

theorem signed_tile_observations (c : SignedViews) (start : Nat)
    (s t : BlockState) (h : ObsEq s t) :
    ObsEq (finalState c.inputView c.outputView c.n start c.B s)
      (finalState c.inputView c.outputView c.n start c.B t) :=
  SourceBranch.tile_obs_congr c.inputView c.outputView c.n start c.B h

def kernelStarts (c : SignedViews) (oneTile : Bool) (tiles : Nat) (s : BlockState) : List Nat :=
  if oneTile then [s.pids 0 * c.B]
  else (List.range tiles).map (fun j => (s.pids 0 + j * s.numPids 0) * c.B)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceGridExecution
