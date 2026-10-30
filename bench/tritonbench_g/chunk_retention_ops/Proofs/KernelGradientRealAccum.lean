import bench.tritonbench_g.chunk_retention_ops.Proofs.KernelGradientRealLoads
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
open VeriTile Triton FP.Structural
open KernelBackward (PID)
open KernelGradient
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

theorem accumulated_query (c : Shape) (pid : PID) (m : Memory ℝ) (scale : ℝ) (seed terminal : Bool)
    (n : Nat) (i : TileIndex [64,c.BK]) :
    (gradStates algebra dotPrimitive (parameters c) pid m scale n).query i=
      blockChain realOps c.BV
        (fun j=>gAt realOps c (inputs c m seed terminal) pid.z (pid.y*64+i.1.val) j)
        (fun j=>masked realOps c.K c.V (history c pid m) (pid.x*c.BK+i.2.1.val) j) n 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change (gradStates algebra dotPrimitive (parameters c) pid m scale n).query i+
      (∑ j : Fin c.BV,valueTile algebra (parameters c) pid m "incoming" n (i.1,j,())*
        gradHistory algebra (parameters c) pid m n (j,i.2.1,()))=_
    rw [ih,blockChain,RealScalar.chain_sum]
    congr 1
    rw [← Fin.sum_univ_eq_sum_range]
    apply Finset.sum_congr rfl
    intro j _
    rw [grad_incoming_load c pid m seed terminal,grad_history_load]

theorem accumulated_key (c : Shape) (pid : PID) (m : Memory ℝ) (scale : ℝ) (seed terminal : Bool)
    (n : Nat) (i : TileIndex [64,c.BK]) :
    (gradStates algebra dotPrimitive (parameters c) pid m scale n).key i=
      blockChain realOps c.BV
        (fun j=>vAt realOps c (inputs c m seed terminal) pid.z (pid.y*64+i.1.val) j)
        (fun j=>masked realOps c.K c.V (dhistory c pid m) (pid.x*c.BK+i.2.1.val) j) n 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change (gradStates algebra dotPrimitive (parameters c) pid m scale n).key i+
      (∑ j : Fin c.BV,valueTile algebra (parameters c) pid m "v" n (i.1,j,())*
        gradHistoryGradient algebra (parameters c) pid m n (i.2.1,j,()))=_
    rw [ih,blockChain,RealScalar.chain_sum]
    congr 1
    rw [← Fin.sum_univ_eq_sum_range]
    apply Finset.sum_congr rfl
    intro j _
    rw [grad_value_load c pid m seed terminal,grad_dhistory_load]

theorem accumulated_score (c : Shape) (pid : PID) (m : Memory ℝ) (scale : ℝ) (seed terminal : Bool)
    (n : Nat) (i : TileIndex [64,64]) :
    (gradStates algebra dotPrimitive (parameters c) pid m scale n).score i=
      blockChain realOps c.BV
        (fun j=>gAt realOps c (inputs c m seed terminal) pid.z (pid.y*64+i.1.val) j)
        (fun j=>vAt realOps c (inputs c m seed terminal) pid.z (pid.y*64+i.2.1.val) j) n 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change (gradStates algebra dotPrimitive (parameters c) pid m scale n).score i+
      (∑ j : Fin c.BV,valueTile algebra (parameters c) pid m "incoming" n (i.1,j,())*
        valueTile algebra (parameters c) pid m "v" n (i.2.1,j,()))=_
    rw [ih,blockChain,RealScalar.chain_sum]
    congr 1
    rw [← Fin.sum_univ_eq_sum_range]
    apply Finset.sum_congr rfl
    intro j _
    rw [grad_incoming_load c pid m seed terminal,grad_value_load c pid m seed terminal]

#axiomsClean accumulated_query
#axiomsClean accumulated_key
#axiomsClean accumulated_score
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
