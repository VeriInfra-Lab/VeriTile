import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Reference
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.QueryProjection
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.KeyValueProjection
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.ScaleReverseReal

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
open scoped BigOperators
noncomputable section

theorem real_scalars_ops (σ : ℝ) : realScalars.ops σ = Math.scaleOps σ := rfl

theorem forward_values_real (shape : Shape) (σ : ℝ) (s : State ℝ)
    (r : Fin 128) (v : Fin shape.BV)
    (hk : (launchTile shape s).key=0) (fits : shape.K≤ shape.BK)
    (ht : (launchTile shape s).chunk*128+r.val<shape.T)
    (hv : (launchTile shape s).value*shape.BV+v.val<shape.V) :
    let values := forwardValues (scalarAlgebra realScalars) (scalarDotPrimitive realScalars) (forwardConfig shape σ) s
    (values.output ⟨r,v,()⟩,values.normalizer ⟨r,()⟩) =
      (outputSpec shape.T shape.K σ (headInputs (inputReaders shape s) (launchTile shape s).bh)
        ((launchTile shape s).chunk*128+r.val) ((launchTile shape s).value*shape.BV+v.val),
       normalizerSpec shape.T shape.K σ (headInputs (inputReaders shape s) (launchTile shape s).bh)
        ((launchTile shape s).chunk*128+r.val)) := by
  have h := forward_scalar_values realScalars shape σ s r v
  rw [real_scalars_ops] at h
  exact h.trans (Math.forward_whole_real (σ := σ) sourcePlans shape (inputReaders shape s)
    (launchTile shape s) r.val v.val hk fits r.isLt ht hv)

theorem query_values_real (shape : Shape) (σ : ℝ) (tile : Tile) (s : State ℝ)
    (r : Fin 64) (e : Fin shape.BK) (hk : tile.key=0) (fits : shape.K≤ shape.BK)
    (ht : tile.chunk*64+r.val<shape.T) (he : e.val<shape.K) :
    queryValues (scalarAlgebra realScalars) (scalarDotPrimitive realScalars) (dqConfig shape σ tile) s ⟨r,e,()⟩ =
      queryPartial shape σ (inputReaders shape s) tile.bh tile.value (tile.chunk*64+r.val) e.val := by
  have h := query_scalar_values realScalars shape σ tile s r e
  rw [real_scalars_ops] at h
  exact h.trans (Math.query_whole_real (σ := σ) sourcePlans shape (inputReaders shape s)
    tile r.val e.val hk fits r.isLt ht he)

theorem keyValue_values_real (shape : Shape) (σ : ℝ) (tile : Tile) (s : State ℝ)
    (r : Fin 64) (e : Fin shape.BK) (v : Fin shape.BV) (hk : tile.key=0) (fits : shape.K≤ shape.BK)
    (hn : tile.chunk*64+r.val<shape.T) (he : e.val<shape.K) (hv : tile.value*shape.BV+v.val<shape.V) :
    let values := backwardValues (scalarAlgebra realScalars) (scalarDotPrimitive realScalars) (backwardConfig shape σ tile) s
    (values.key ⟨r,e,()⟩,values.value ⟨r,v,()⟩) =
      (keyPartial shape σ (inputReaders shape s) tile.bh tile.value (tile.chunk*64+r.val) e.val,
       valueGradient shape σ (inputReaders shape s) tile.bh (tile.chunk*64+r.val) (tile.value*shape.BV+v.val)) := by
  have h := keyValue_scalar_values realScalars shape σ tile s r e v
  rw [real_scalars_ops] at h
  exact h.trans (Math.keyValue_whole_real (σ := σ) sourcePlans shape (inputReaders shape s)
    tile r.val e.val v.val hk fits hn he hv r.isLt)

#axiomsClean real_scalars_ops
#axiomsClean forward_values_real
#axiomsClean query_values_real
#axiomsClean keyValue_values_real
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
