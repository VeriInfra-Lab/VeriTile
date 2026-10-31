import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_StructuralControl
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_StructuralTile
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_HostPlan

/-!
Actual selected-FP32 source execution on an opaque floating carrier, from the
original wrapper preparation through both constexpr branches and all CTAs.
The local source, loop and launch proofs are discharged, and outputs use the
entry memory. Scalar casts and strict comparison remain explicit. Specializing
the algebra to raw FP32 words does not change this source or its control flow.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.StructuralHost
open VeriTile.Triton FP.Structural SignedView SignedGrid HostPlan
open StructuralGrid StructuralControl StructuralTile SourceSignedBranch
noncomputable section

theorem tile_contract {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (c : SignedViews) {capacity : Nat} (hout : c.outputView.Valid c.n capacity)
    (hstride : c.outputView.stride ≠ 0) :
    TileContract c.inputView c.outputView c.n c.B
      (fun x => Cell.mk .fp32 (scalarResult M cmp x)) (flatTile (tileOutput M cmp c)) := by
  intro T s
  refine ⟨flatState (tileOutput M cmp c) T s, rfl, ?_, ?_⟩
  · exact fun lane hactive => tileOutput_readback M cmp c T s.mem hout hstride lane hactive
  · exact fun r a outside => tileOutput_frame M cmp c T s.mem r a outside

def kernelIds {α : Type} (oneTile : Bool) (tiles : Nat) (s : State α) : List Nat :=
  if oneTile then [s.pids 0]
  else (List.range tiles).map (fun j => s.pids 0 + j * s.numPids 0)

theorem prepared_ids (input output : Buffer) (n : Nat) (sx sy : Int) :
    sourceTileIds (rankOneArguments input output n sx sy) = cappedTileIds n (block n) := by
  change (List.range (CappedGrid.numCTAs n (block n))).flatMap
    (fun pid => idsPerProgram (rankOneArguments input output n sx sy) pid) = _
  by_cases hone : CappedGrid.tilesPerCTA n (block n) = 1
  · simp [idsPerProgram, rankOneArguments, hone, cappedTileIds]
  · simp [idsPerProgram, rankOneArguments, hone, cappedTileIds]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.StructuralHost
