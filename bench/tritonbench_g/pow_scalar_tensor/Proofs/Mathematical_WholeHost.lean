import bench.tritonbench_g.pow_scalar_tensor.Proofs.Mathematical_WholeTile

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.WholeHost
open VeriTile.Triton FP.Structural WholeSource WholeTile
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open SignedView SignedGrid HostPlan StructuralGrid StructuralControl
noncomputable section

theorem tile_contract {α : Type} [Inhabited α] (M : Algebra α) (c : Config) {capacity : Nat} (hout : c.outputView.Valid c.n capacity)
    (hstride : c.outputView.stride ≠ 0) :
    TileContract c.inputView c.outputView c.n c.B
      (fun x => Cell.mk .fp32 (scalarResult M c.base x)) (flatTile (tileOutput M c)) := by
  intro T s
  refine ⟨flatState (tileOutput M c) T s, rfl, ?_, ?_⟩
  · exact fun lane hactive => tileOutput_readback M c T s.mem hout hstride lane hactive
  · exact fun r a outside => tileOutput_frame M c T s.mem r a outside

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
end VeriTile.Bench.TritonBenchG.PowScalarTensor.WholeHost
