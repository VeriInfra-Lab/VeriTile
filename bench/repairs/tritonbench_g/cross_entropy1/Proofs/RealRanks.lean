import bench.repairs.tritonbench_g.cross_entropy1.Proofs.RealHost

namespace VeriTile.Bench.Repairs.CrossEntropy1.RealHost
open VeriTile.Triton Source Host
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000

theorem rank_sum_value (world rows row rank : Nat) (positive : 0<world) (rankBound : rank<world)
    (own peers : Nat → WithBot ℝ) (f : Nat → ℝ) (ownValue : own row=↑(f rank))
    (sent : ∀ r, r<world → peers (r*rows+row)=↑(f r)) :
    rankResult arithmetic.rankSum world rows own peers row=↑(∑ r∈Finset.range world, f r) := by
  by_cases many : world>1
  · rw [rankResult,if_pos many,mapped_slots world rows row peers f sent]
    exact sum_range world f
  · have one : world=1 := by omega
    have zero : rank=0 := by omega
    simpa [rankResult,one,zero] using ownValue

theorem rank_lse_value (world rows row rank : Nat) (positive : 0<world) (rankBound : rank<world)
    (own peers : Nat → WithBot ℝ) (f : Nat → ℝ) (ownValue : own row=↑(f rank))
    (sent : ∀ r, r<world → peers (r*rows+row)=↑(f r)) :
    rankResult arithmetic.logsumexp world rows own peers row=↑(Real.log (∑ r∈Finset.range world, Real.exp (f r))) := by
  by_cases many : world>1
  · rw [rankResult,if_pos many,mapped_slots world rows row peers f sent]
    exact logsumexp_range world f
  · have one : world=1 := by omega
    have zero : rank=0 := by omega
    simpa [rankResult,one,zero,Real.log_exp] using ownValue

theorem correction_real (scale loss lse : ℝ) :
    correctedValue arithmetic scale (↑loss) (↑lse)=↑(loss+(if scale≠0 then scale*lse^2 else 0)) := by
  by_cases nonzero : scale≠0
  · simp only [correctedValue,if_pos nonzero]
    rfl
  · simp only [correctedValue,if_neg nonzero,add_zero]

/-- The final scalar expression, after actual rank-local tensors have supplied
the peer inputs. This lemma has no runtime or execution-success premise. -/
theorem rank_loss_math (o : Options) (world cols rows row rank : Nat) (positive : 0<world)
    (colsPositive : 0<cols) (rankBound : rank<world) (target : Int) (x : Nat → ℝ)
    (valid : target=o.ignored ∨ (0≤target ∧ target<(world*cols : Nat)))
    (ownLoss ownLSE lossPeers lsePeers : Nat → WithBot ℝ)
    (localLoss : ownLoss row=↑(if target=o.ignored then 0 else
      Aggregation.rankContribution world cols target.toNat o.smoothing x rank))
    (localLSE : ownLSE row=↑(Math.lse cols (fun col => x (rank*cols+col))))
    (sentLoss : ∀ r, r<world → lossPeers (r*rows+row)=↑(if target=o.ignored then 0 else
      Aggregation.rankContribution world cols target.toNat o.smoothing x r))
    (sentLSE : ∀ r, r<world → lsePeers (r*rows+row)=↑(Math.lse cols (fun col => x (r*cols+col)))) :
    (if target=o.ignored then arithmetic.zero else
      correctedValue arithmetic o.squareScale
        (arithmetic.add (rankResult arithmetic.rankSum world rows ownLoss lossPeers row)
          (rankResult arithmetic.logsumexp world rows ownLSE lsePeers row))
        (rankResult arithmetic.logsumexp world rows ownLSE lsePeers row))=
      ↑(Math.loss (world*cols) x target o.ignored o.smoothing o.squareScale) ∧
    rankResult arithmetic.logsumexp world rows ownLSE lsePeers row=↑(Math.lse (world*cols) x) := by
  have sums := rank_sum_value world rows row rank positive rankBound ownLoss lossPeers
    (fun r => if target=o.ignored then 0 else Aggregation.rankContribution world cols target.toNat o.smoothing x r) localLoss sentLoss
  have lse := rank_lse_value world rows row rank positive rankBound ownLSE lsePeers
    (fun r => Math.lse cols (fun col => x (r*cols+col))) localLSE sentLSE
  rw [Math.rank_lse_composition world cols colsPositive x] at lse
  refine ⟨?_,lse⟩
  by_cases ignored : target=o.ignored
  · simp only [ignored,↓reduceIte,arithmetic,Math.loss]
  · rw [if_neg ignored,sums,lse]
    simp only [ignored,↓reduceIte] 
    change correctedValue arithmetic o.squareScale
      (↑((∑ r∈Finset.range world, Aggregation.rankContribution world cols target.toNat o.smoothing x r)+Math.lse (world*cols) x))
      (↑(Math.lse (world*cols) x))=_
    rw [correction_real]
    apply congrArg (fun r : ℝ => (↑r : WithBot ℝ))
    have math := Aggregation.global_loss world cols colsPositive target o.ignored valid o.smoothing o.squareScale x
    simpa only [if_neg ignored] using math

#axiomsClean rank_sum_value
#axiomsClean rank_lse_value
#axiomsClean rank_loss_math
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.RealHost
