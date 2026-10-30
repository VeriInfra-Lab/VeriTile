import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelGradientRealAccum
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
open VeriTile Triton FP.Structural
open KernelBackward (PID)
open KernelGradient
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def accumulated (c : Shape) (pid : PID) (m : Memory ℝ) (scale : ℝ) :=
  gradStates algebra dotPrimitive (parameters c) pid m scale c.valueBlocks

theorem score_gradient_eq (c : Shape) (pid : PID) (m : Memory ℝ) (scale : ℝ)
    (seed terminal : Bool) (i : TileIndex [64,64]) :
    gradScoreGradient algebra (parameters c) pid scale (accumulated c pid m scale) i=
      Body.ds realOps c (inputs c m seed terminal) scale pid.z pid.y i.1.val i.2.1.val := by
  change (gradStates algebra dotPrimitive (parameters c) pid m scale c.valueBlocks).score i*
    scoreDecay algebra (parameters c) pid scale i=_
  rw [accumulated_score c pid m scale seed terminal,score_decay_eq]
  rfl

theorem partial_score_eq (c : Shape) (pid : PID) (m : Memory ℝ) (scale : ℝ)
    (seed terminal : Bool) (i : TileIndex [64,64]) :
    gradScore algebra dotPrimitive (parameters c) pid m scale i=
      Body.partialScore realOps c (inputs c m seed terminal) scale pid.x pid.z pid.y i.1.val i.2.1.val := by
  change (∑ e : Fin c.BK,gradKey algebra (parameters c) pid m (i.1,e,())*
    gradQuery algebra (parameters c) pid m (e,i.2.1,()))*
      scoreDecay algebra (parameters c) pid scale (i.2.1,i.1,())=_
  rw [score_decay_eq]
  unfold Body.partialScore
  rw [RealScalar.chain_sum]
  simp only [RealScalar.real_mul,RealScalar.real_zero,zero_add]
  congr 1
  rw [← Fin.sum_univ_eq_sum_range]
  apply Finset.sum_congr rfl
  intro e _
  rw [grad_key_load c pid m seed terminal,grad_query_load c pid m seed terminal]

theorem query_result_eq (c : Shape) (pid : PID) (m : Memory ℝ) (scale : ℝ)
    (seed terminal : Bool) (i : TileIndex [64,c.BK]) :
    gradQueryResult algebra dotPrimitive (parameters c) pid m scale (accumulated c pid m scale) i=
      Body.dq realOps c (inputs c m seed terminal) scale pid.z pid.y (history c pid m)
        i.1.val (pid.x*c.BK+i.2.1.val) := by
  change (gradStates algebra dotPrimitive (parameters c) pid m scale c.valueBlocks).query i*
    queryDecay algebra (parameters c) pid scale i.1.val+
    (∑ t : Fin 64,gradScoreGradient algebra (parameters c) pid scale (accumulated c pid m scale) (i.1,t,())*
      gradKey algebra (parameters c) pid m (t,i.2.1,()))=_
  rw [accumulated_query c pid m scale seed terminal,query_decay_eq]
  unfold Body.dq
  rw [RealScalar.chain_sum]
  simp only [RealScalar.real_cast,RealScalar.real_mul]
  congr 1
  rw [← Fin.sum_univ_eq_sum_range]
  apply Finset.sum_congr rfl
  intro t _
  rw [score_gradient_eq c pid m scale seed terminal,grad_key_load c pid m seed terminal]

theorem key_result_eq (c : Shape) (pid : PID) (m : Memory ℝ) (scale : ℝ)
    (seed terminal : Bool) (i : TileIndex [64,c.BK]) :
    gradKeyResult algebra dotPrimitive (parameters c) pid m scale (accumulated c pid m scale) i=
      Body.dk realOps c (inputs c m seed terminal) scale pid.z pid.y (dhistory c pid m)
        i.1.val (pid.x*c.BK+i.2.1.val) := by
  change (gradStates algebra dotPrimitive (parameters c) pid m scale c.valueBlocks).key i*
    keyDecay algebra (parameters c) pid i.1.val+
    (∑ r : Fin 64,gradQuery algebra (parameters c) pid m (i.2.1,r,())*
      gradScoreGradient algebra (parameters c) pid scale (accumulated c pid m scale) (r,i.1,()))=_
  rw [accumulated_key c pid m scale seed terminal,key_decay_eq]
  unfold Body.dk
  rw [RealScalar.chain_sum]
  simp only [RealScalar.real_cast,RealScalar.real_mul,RealScalar.real_add,RealScalar.real_zero,zero_add]
  congr 1
  rw [← Fin.sum_univ_eq_sum_range]
  apply Finset.sum_congr rfl
  intro r _
  rw [grad_query_load c pid m seed terminal,score_gradient_eq c pid m scale seed terminal]

theorem value_result_eq (c : Shape) (pid : PID) (m : Memory ℝ) (scale : ℝ)
    (seed terminal : Bool) (b : Nat) (i : TileIndex [64,c.BV]) :
    gradValue algebra dotPrimitive (parameters c) pid m scale b i=
      Body.dvPartial realOps c (inputs c m seed terminal) scale pid.x pid.z pid.y (dhistory c pid m)
        i.1.val (b*c.BV+i.2.1.val) := by
  change (∑ e : Fin c.BK,gradKey algebra (parameters c) pid m (i.1,e,())*
      gradHistoryGradient algebra (parameters c) pid m b (e,i.2.1,()))*
    keyDecay algebra (parameters c) pid i.1.val+
    (∑ r : Fin 64,gradScore algebra dotPrimitive (parameters c) pid m scale (i.1,r,())*
      valueTile algebra (parameters c) pid m "incoming" b (r,i.2.1,()))=_
  rw [key_decay_eq]
  unfold Body.dvPartial
  rw [RealScalar.chain_sum,RealScalar.chain_sum]
  simp only [RealScalar.real_cast,RealScalar.real_mul,RealScalar.real_zero,zero_add]
  congr 1
  · congr 1
    rw [← Fin.sum_univ_eq_sum_range]
    apply Finset.sum_congr rfl
    intro e _
    rw [grad_key_load c pid m seed terminal,grad_dhistory_load]
  · rw [← Fin.sum_univ_eq_sum_range]
    apply Finset.sum_congr rfl
    intro r _
    rw [partial_score_eq c pid m scale seed terminal,grad_incoming_load c pid m seed terminal]

#axiomsClean score_gradient_eq
#axiomsClean partial_score_eq
#axiomsClean query_result_eq
#axiomsClean key_result_eq
#axiomsClean value_result_eq
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
