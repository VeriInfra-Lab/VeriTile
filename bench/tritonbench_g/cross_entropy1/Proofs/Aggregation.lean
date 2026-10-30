import bench.tritonbench_g.cross_entropy1.Proofs.Real

namespace VeriTile.Bench.Repairs.CrossEntropy1.Aggregation
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000

theorem target_partition (classes width : Nat) (target : Int) (positive : 0<width) (value : ℝ) :
    (∑ block∈Finset.range (Math.blocks classes width),
      if (block*width : Int)≤target ∧ target<(min classes ((block+1)*width) : Nat) then value else 0)=
      if 0≤target ∧ target<(classes : Int) then value else 0 := by
  by_cases nonneg : 0≤target
  · have each : ∀ block : Nat,
        ((block*width : Int)≤target ∧ target<(min classes ((block+1)*width) : Nat)) ↔
          block*width≤target.toNat ∧ target.toNat<min classes ((block+1)*width) := by
      intro block
      omega
    simp_rw [each]
    rw [Math.partition_target classes width target.toNat positive value]
    have cond : target.toNat<classes ↔ 0≤target ∧ target<(classes : Int) := by omega
    exact if_congr cond rfl rfl
  · have each : ∀ block : Nat, ¬((block*width : Int)≤target ∧ target<(min classes ((block+1)*width) : Nat)) := by
      intro block
      have bound : (0 : Int)≤(block*width : Nat) := Int.natCast_nonneg _
      omega
    simp only [each,nonneg,false_and,if_false,Finset.sum_const_zero]

theorem local_split_sum (p : Parameters) (x : Nat → ℝ) (target : Int) (positive : 0<p.block) :
    (∑ block∈Finset.range (Math.blocks p.cols p.block), RealSource.partialLoss p x target block)=
      if target=p.ignored then 0 else
        -p.smoothing*(∑ j∈Finset.range p.cols, x j)/p.totalClasses-
          if 0≤target-p.classStart ∧ target-p.classStart<(p.cols : Int) then
            (1-p.smoothing)*x (target-p.classStart).toNat else 0 := by
  by_cases ignored : target=p.ignored
  · simp [RealSource.partialLoss,ignored]
  · simp only [RealSource.partialLoss,ignored,if_false,Finset.sum_sub_distrib]
    simp only [div_eq_mul_inv]
    rw [← Finset.sum_mul,← Finset.mul_sum,Math.partition p.cols p.block positive]
    rw [target_partition p.cols p.block (target-p.classStart) positive]

theorem blocks_product (ranks classes : Nat) (positive : 0<classes) :
    Math.blocks (ranks*classes) classes=ranks := by
  unfold Math.blocks
  apply Nat.div_eq_of_lt_le
  · omega
  · rw [Nat.add_mul,Nat.one_mul]
    omega

/-- Global target ownership is independent of local split count. -/
theorem rank_target (ranks classes target : Nat) (positive : 0<classes) (value : ℝ) :
    (∑ rank∈Finset.range ranks,
      if rank*classes≤target ∧ target<rank*classes+classes then value else 0)=
      if target<ranks*classes then value else 0 := by
  have source := Math.partition_target (ranks*classes) classes target positive value
  rw [blocks_product ranks classes positive] at source
  have each : ∀ rank∈Finset.range ranks,
      (rank*classes≤target ∧ target<min (ranks*classes) ((rank+1)*classes)) ↔
        rank*classes≤target ∧ target<rank*classes+classes := by
    intro rank hr
    have bound := Nat.mul_le_mul_right classes (Nat.succ_le_of_lt (Finset.mem_range.mp hr))
    rw [Nat.succ_mul] at bound
    rw [Nat.add_mul,Nat.one_mul,Nat.min_eq_right bound]
  rw [Finset.sum_congr rfl (fun rank hr => if_congr (each rank hr) rfl rfl)] at source
  exact source

/-- The two host logsumexp operations compose the actual local block LSEs
into the independent global row LSE. -/
theorem nested_lse (ranks classes width : Nat) (positive : 0<classes) (widthPositive : 0<width)
    (x : Nat → ℝ) :
    Real.log (∑ rank∈Finset.range ranks,
      Real.exp (Real.log (∑ block∈Finset.range (Math.blocks classes width),
        Real.exp (Math.blockLSE classes width block (fun j => x (rank*classes+j))))))=
      Math.lse (ranks*classes) x := by
  simp_rw [Math.local_lse_composition classes width widthPositive]
  exact Math.rank_lse_composition ranks classes positive x

def rankContribution (ranks classes target : Nat) (smoothing : ℝ) (x : Nat → ℝ) (rank : Nat) : ℝ :=
  -smoothing*(∑ j∈Finset.range classes, x (rank*classes+j))/(ranks*classes)-
    if rank*classes≤target ∧ target<rank*classes+classes then (1-smoothing)*x target else 0

theorem rank_sum (ranks classes target : Nat) (positive : 0<classes)
    (valid : target<ranks*classes) (smoothing : ℝ) (x : Nat → ℝ) :
    (∑ rank∈Finset.range ranks, rankContribution ranks classes target smoothing x rank)=
      -smoothing*(∑ j∈Finset.range (ranks*classes), x j)/(ranks*classes)-(1-smoothing)*x target := by
  simp only [rankContribution,Finset.sum_sub_distrib,div_eq_mul_inv]
  rw [← Finset.sum_mul,← Finset.mul_sum,Math.sum_blocks ranks classes x,
    rank_target ranks classes target positive,if_pos valid]

/-- Independent full-row mathematical loss after ordinary rank SUM, LSE
addition, the optional square correction, and the final ignored-row mask. -/
theorem global_loss (ranks classes : Nat) (positive : 0<classes) (target ignored : Int)
    (valid : target=ignored ∨ (0≤target ∧ target<(ranks*classes : Nat)))
    (smoothing scale : ℝ) (x : Nat → ℝ) :
    (if target=ignored then 0 else
      (∑ rank∈Finset.range ranks, rankContribution ranks classes target.toNat smoothing x rank)+
      Math.lse (ranks*classes) x+
      (if scale≠0 then scale*(Math.lse (ranks*classes) x)^2 else 0))=
      Math.loss (ranks*classes) x target ignored smoothing scale := by
  by_cases ignore : target=ignored
  · simp [ignore,Math.loss]
  · have targetValid := valid.resolve_left ignore
    have targetBound : target.toNat<ranks*classes := by omega
    simp only [ignore,if_false,Math.loss,rank_sum ranks classes target.toNat positive targetBound]
    by_cases hs : scale=0
    · simp [hs]
      ring
    · rw [if_pos hs]
      push_cast
      ring

def rankParameters (p : Parameters) (ranks rank : Nat) : Parameters :=
  {p with totalClasses:=ranks*p.cols, classStart:=(rank*p.cols : Nat)}

theorem actual_rank_contribution (p : Parameters) (ranks rank : Nat) (x : Nat → ℝ)
    (target : Int) (nonneg : 0≤target) (notIgnored : target≠p.ignored) (positive : 0<p.block) :
    (∑ block∈Finset.range (Math.blocks p.cols p.block),
      RealSource.partialLoss (rankParameters p ranks rank) (fun j => x (rank*p.cols+j)) target block)=
      rankContribution ranks p.cols target.toNat p.smoothing x rank := by
  refine (local_split_sum (rankParameters p ranks rank) (fun j => x (rank*p.cols+j)) target positive).trans ?_
  simp only [rankParameters,notIgnored,if_false,rankContribution]
  have cond : (0≤target-(rank*p.cols : Nat) ∧ target-(rank*p.cols : Nat)<(p.cols : Int)) ↔
      rank*p.cols≤target.toNat ∧ target.toNat<rank*p.cols+p.cols := by omega
  by_cases inside : rank*p.cols≤target.toNat ∧ target.toNat<rank*p.cols+p.cols
  · have shifted : rank*p.cols+(target-(rank*p.cols : Nat)).toNat=target.toNat := by omega
    simp only [if_pos inside,if_pos (cond.mpr inside),shifted]
    push_cast
    rfl
  · simp only [if_neg inside,if_neg (fun h => inside (cond.mp h))]
    push_cast
    rfl

#axiomsClean global_loss
#axiomsClean actual_rank_contribution
#axiomsClean local_split_sum
#axiomsClean rank_target
#axiomsClean nested_lse
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Aggregation
