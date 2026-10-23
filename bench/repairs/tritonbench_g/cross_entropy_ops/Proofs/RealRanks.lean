import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Correction
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Math
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.RealRanks

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open CrossEntropy1.Host (TensorArithmetic rankResult)
open CrossEntropy1.RealHost (arithmetic)
noncomputable section
set_option maxHeartbeats 8000000

theorem masked_correction {α : Type} (A : TensorArithmetic α) (o : Options) (target : Int) (loss lse : α) :
    (if target=o.ignored then A.zero else correctionLoss A o target loss lse)=
      (if target=o.ignored then A.zero else CrossEntropy1.Host.correctedValue A o.squareScale loss lse) := by
  by_cases ignored : target=o.ignored <;> by_cases nonzero : o.squareScale≠0 <;>
    simp [correctionLoss,correctionZ,CrossEntropy1.Host.correctedValue,ignored,nonzero]

theorem correction_z_real (o : Options) (target : Int) (L : ℝ) :
    correctionZ arithmetic o target (↑L)=↑(if target=o.ignored then 0 else o.squareScale*L*L) := by
  by_cases nonzero : o.squareScale≠0
  · by_cases ignored : target=o.ignored
    · simp only [correctionZ,if_pos nonzero,if_pos ignored]
      rfl
    · simp only [correctionZ,if_pos nonzero,if_neg ignored]
      change (↑(o.squareScale*L^2) : WithBot ℝ)=↑(o.squareScale*L*L)
      congr 1
      ring
  · have zero : o.squareScale=0 := not_not.mp nonzero
    simp [correctionZ,zero,arithmetic]

/-- Independent global Real formulas derived from local shard formulas.
This statement has no execution or returned-value premise. -/
theorem rank_loss_math (o : Options) (world cols rows row rank : Nat) (positive : 0<world)
    (colsPositive : 0<cols) (rankBound : rank<world) (target : Int) (x : Nat → ℝ)
    (valid : target=o.ignored ∨ (0≤target ∧ target<(world*cols : Nat)))
    (ownLoss ownLSE lossPeers lsePeers : Nat → WithBot ℝ)
    (localLoss : ownLoss row=↑(if target=o.ignored then 0 else
      CrossEntropy1.Aggregation.rankContribution world cols target.toNat o.smoothing (Spec.scaled o.logitScale x) rank))
    (localLSE : ownLSE row=↑(Spec.lse cols o.logitScale (fun col => x (rank*cols+col))))
    (sentLoss : ∀ r, r<world → lossPeers (r*rows+row)=↑(if target=o.ignored then 0 else
      CrossEntropy1.Aggregation.rankContribution world cols target.toNat o.smoothing (Spec.scaled o.logitScale x) r))
    (sentLSE : ∀ r, r<world → lsePeers (r*rows+row)=↑(Spec.lse cols o.logitScale (fun col => x (r*cols+col)))) :
    (if target=o.ignored then arithmetic.zero else
      correctionLoss arithmetic o target
        (arithmetic.add (rankResult arithmetic.rankSum world rows ownLoss lossPeers row)
          (rankResult arithmetic.logsumexp world rows ownLSE lsePeers row))
        (rankResult arithmetic.logsumexp world rows ownLSE lsePeers row))=
      ↑(Spec.loss (world*cols) o.logitScale o.smoothing o.squareScale x target o.ignored) ∧
    rankResult arithmetic.logsumexp world rows ownLSE lsePeers row=↑(Spec.lse (world*cols) o.logitScale x) ∧
    correctionZ arithmetic o target (rankResult arithmetic.logsumexp world rows ownLSE lsePeers row)=
      ↑(Spec.zLoss (world*cols) o.logitScale o.squareScale x target o.ignored) := by
  have base := CrossEntropy1.RealHost.rank_loss_math o.toOptions world cols rows row rank positive colsPositive rankBound
    target (Spec.scaled o.logitScale x) valid ownLoss ownLSE lossPeers lsePeers localLoss localLSE sentLoss sentLSE
  refine ⟨?_,base.2,?_⟩
  · rw [masked_correction]
    exact base.1
  · rw [base.2,correction_z_real]
    rfl

#axiomsClean masked_correction
#axiomsClean correction_z_real
#axiomsClean rank_loss_math
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
