import bench.tritonbench_g.parallel_attention.QueryScalar
import bench.tritonbench_g.parallel_attention.ForwardProjection

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

theorem query_scalar_values {α : Type} [Inhabited α] (a : Scalars α)
    (shape : Shape) (scale : ℝ) (tile : Tile) (s : State α) (r : Fin 64) (e : Fin shape.BK) :
    a.cast (queryValues (scalarAlgebra a) (scalarDotPrimitive a) (dqConfig shape scale tile) s ⟨r,e,()⟩) =
      queryLane (a.ops scale) sourcePlans shape (inputReaders shape s) tile r.val e.val := by
  let c := dqConfig shape scale tile
  let x := inputReaders shape s
  let observe := fun (z : TileIndex [64,shape.BK] → α) => z ⟨r,e,()⟩
  let update := queryUpdate (scalarAlgebra a) (scalarDotPrimitive a) c s
  let step := queryStep (a.ops scale) sourcePlans shape x tile r.val e.val
  have hprefix (initial : TileIndex [64,shape.BK] → α) :
      observe (foldRange (update Bool.false) 0 (tile.chunk*64) 32 initial) =
        foldN (step Bool.false) (tile.chunk*2) (observe initial) := by
    have bound : tile.chunk*64 = 0+(tile.chunk*2)*32 := by omega
    rw [bound,foldRange_count _ _ _ _ (by omega)]
    apply foldN_projection
    intro j z
    simpa [observe,update,step,c,x] using query_scalar_step a shape scale tile s Bool.false j r e z
  have diagonal (initial : TileIndex [64,shape.BK] → α) :
      observe (foldRange (update Bool.true) (tile.chunk*64) ((tile.chunk+1)*64) 32 initial) =
        foldN (step Bool.true) 2 (observe initial) := by
    have bound : (tile.chunk+1)*64 = tile.chunk*64+2*32 := by omega
    rw [bound,foldRange_count _ _ _ _ (by omega)]
    apply foldN_projection
    intro j z
    exact query_scalar_step a shape scale tile s Bool.true j r e z
  let prior := foldRange (update Bool.false) 0 (tile.chunk*64) 32 (fun _ => a.literal 0)
  have hp : observe prior = foldN (step Bool.false) (tile.chunk*2) (a.literal 0) := hprefix _
  have hd := diagonal (queryScale (scalarAlgebra a) c prior)
  have hs : observe (queryScale (scalarAlgebra a) c prior)=
      a.mul (foldN (step Bool.false) (tile.chunk*2) (a.literal 0)) (a.literal scale) :=
    congrArg (fun z => a.mul z (a.literal scale)) hp
  rw [hs] at hd
  exact congrArg a.cast hd

#axiomsClean query_scalar_values
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
