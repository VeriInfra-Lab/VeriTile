import bench.tritonbench_g.parallel_attention.KeyValueScalar
import bench.tritonbench_g.parallel_attention.ForwardProjection

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

theorem descending_count_exact (shape : Shape) (scale : ℝ) (tile : Tile) :
    descendingCount (backwardConfig shape scale tile)=blocks shape.T 32-(tile.chunk+1)*2 := by
  simp only [descendingCount,backwardConfig,forwardConfig,blocks]
  omega

theorem descending_start_exact (shape : Shape) (scale : ℝ) (tile : Tile) (j : Nat) :
    descendingStart (backwardConfig shape scale tile) j=Math.kvStart shape tile Bool.false j := by
  simp [descendingStart,backwardConfig,forwardConfig,Math.kvStart,blocks]
  omega

theorem keyValue_scalar_values {α : Type} [Inhabited α] (a : Scalars α)
    (shape : Shape) (scale : ℝ) (tile : Tile) (s : State α)
    (r : Fin 64) (e : Fin shape.BK) (v : Fin shape.BV) :
    let values := backwardValues (scalarAlgebra a) (scalarDotPrimitive a) (backwardConfig shape scale tile) s
    (a.cast (values.key ⟨r,e,()⟩),a.cast (values.value ⟨r,v,()⟩)) =
      keyValueLane (a.ops scale) sourcePlans shape (inputReaders shape s) tile r.val e.val v.val := by
  let c := backwardConfig shape scale tile
  let x := inputReaders shape s
  let observe := fun (z : BackwardAccumulator α 64 shape.BK shape.BV) => (z.key ⟨r,e,()⟩,z.value ⟨r,v,()⟩)
  let update := backwardUpdate (scalarAlgebra a) (scalarDotPrimitive a) c s
  let step := keyValueStep (a.ops scale) sourcePlans shape x tile r.val e.val v.val
  have future (initial : BackwardAccumulator α 64 shape.BK shape.BV) :
      observe (foldRange (fun j => update Bool.false (descendingStart c j)) 0 (descendingCount c) 1 initial) =
        foldN (step Bool.false) (blocks shape.T 32-(tile.chunk+1)*2) (observe initial) := by
    rw [descending_count_exact]
    have bound : blocks shape.T 32-(tile.chunk+1)*2=0+(blocks shape.T 32-(tile.chunk+1)*2)*1 := by omega
    rw [bound,foldRange_count _ _ _ _ (by omega)]
    simp only [Nat.mul_one,Nat.zero_add]
    apply foldN_projection
    intro j z
    simpa [observe,update,step,c,x,descending_start_exact] using keyValue_scalar_step a shape scale tile s Bool.false j r e v z
  have diagonal (initial : BackwardAccumulator α 64 shape.BK shape.BV) :
      observe (foldRange (update Bool.true) (tile.chunk*64) ((tile.chunk+1)*64) 32 initial) =
        foldN (step Bool.true) 2 (observe initial) := by
    have bound : (tile.chunk+1)*64 = tile.chunk*64+2*32 := by omega
    rw [bound,foldRange_count _ _ _ _ (by omega)]
    apply foldN_projection
    intro j z
    exact keyValue_scalar_step a shape scale tile s Bool.true j r e v z
  let initial : BackwardAccumulator α 64 shape.BK shape.BV := ⟨fun _ => a.literal 0,fun _ => a.literal 0⟩
  have whole := diagonal (foldRange (fun j => update Bool.false (descendingStart c j)) 0 (descendingCount c) 1 initial)
  rw [future] at whole
  exact congrArg (fun z : α×α => (a.cast z.1,a.cast z.2)) whole

#axiomsClean descending_count_exact
#axiomsClean descending_start_exact
#axiomsClean keyValue_scalar_values
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
