import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Engine

namespace VeriTile.Bench.Repairs.LogsumexpFwd.RealModel
open VeriTile.Triton TiledLogSumExp
noncomputable section
open Classical
set_option maxHeartbeats 2000000

/-- Independent mathematical row formula. Scale is applied to each logical
input coordinate before exponentiation. No block/max/reduction schedule
appears here. Empty last axes have the extended value negative infinity. -/
def formula (input : Nat → ℝ) (row D : Nat) (hasScale : Bool) (scale : ℝ) : WithBot ℝ :=
  if D=0 then ⊥ else
    (Real.log (∑ i : Fin D,Real.exp (if hasScale then input (row*D+i.val)*scale
      else input (row*D+i.val))) : ℝ)

def maximum (width : Nat) (values : Fin width → WithBot ℝ) : WithBot ℝ :=
  if h : 0<width then Finset.univ.sup' ⟨⟨0,h⟩,Finset.mem_univ _⟩ values else ⊥

/-- Mathematical interpretation, independent of native words. Neither
nonfinite literal denotes a Real number: both use the non-Real sentinel.
For each launched block the maximum is proved Real, so both infinity tests
are false. Native exceptional behavior is covered separately on raw words. -/
def engine : Engine (WithBot ℝ) where
  zero := (0 : ℝ)
  negInf := ⊥
  bits := fun _ => ⊥
  multiply := Option.map₂ (fun x y : ℝ => x*y)
  subtract := Option.map₂ (fun x y : ℝ => x-y)
  add := Option.map₂ (fun x y : ℝ => x+y)
  exponential := WithBot.realExp
  logarithm := WithBot.realLog
  equal := fun x y => decide (x=y)
  maximum := maximum
  sum := fun _ values => ∑ i,values i

theorem block_real (input : Nat → ℝ) (row block D n : Nat)
    (hasScale : Bool) (scale : ℝ) (active : block*(n+1)<D) :
    blockValue engine (fun i => (input i : WithBot ℝ)) (scale : WithBot ℝ)
      row block D (n+1) hasScale=
      (partialLSE_full (fun i : Fin D => input (row*D+i.val)) block active hasScale scale : WithBot ℝ) := by
  let scaled := fun i : Fin (n+1) => if hasScale then input (row*D+(block*(n+1)+i.val))*scale
    else input (row*D+(block*(n+1)+i.val))
  have hne : (Finset.univ : Finset (Fin (n+1))).Nonempty := Finset.univ_nonempty
  have hv := validLanes_nonempty active
  have values : (fun i : Fin (n+1) => if block*(n+1)+i.val<D then
      if hasScale then engine.multiply (input (row*D+(block*(n+1)+i.val))) scale
        else (input (row*D+(block*(n+1)+i.val)) : WithBot ℝ)
      else engine.negInf)=
    (fun i : Fin (n+1) => if block*(n+1)+i.val<D then (some (scaled i) : WithBot ℝ) else none) := by
    funext i
    by_cases hi : block*(n+1)+i.val<D
    · simp only [if_pos hi]
      cases hasScale <;> rfl
    · simp only [if_neg hi]
      rfl
  unfold blockValue
  rw [values]
  simp only [engine,maximum,Nat.succ_pos,dif_pos]
  erw [sup'_masked_eq hne hv scaled]
  simp only [WithBot.coe_ne_bot,decide_false,Bool.or_false,Bool.false_eq_true,if_false]
  rw [sum_exp_masked_eq scaled]
  simp only [WithBot.realLog_coe]
  erw [Option.map₂_coe_coe]
  apply congrArg (fun x : ℝ => (x : WithBot ℝ))
  rw [add_comm (Real.log _) _]
  unfold partialLSE_full
  have same : ∀ i∈validLanes n D block,
      scaled i=scaledLane_full (fun i : Fin D => input (row*D+i.val)) block hasScale scale i := by
    intro i hi
    have bound := mem_validLanes_iff.mp hi
    simp [scaled,scaledLane_full,bound]
  have hm : (validLanes n D block).sup' hv scaled=
      (validLanes n D block).sup' hv
        (scaledLane_full (fun i : Fin D => input (row*D+i.val)) block hasScale scale) :=
    Finset.sup'_congr hv rfl same
  rw [hm]
  congr 2
  apply Finset.sum_congr rfl
  intro i hi
  rw [same i hi]

theorem aggregate_real (input : Nat → ℝ) (row D n : Nat)
    (hasScale : Bool) (scale : ℝ) (positive : 0<D) :
    (Real.log (∑ block : Fin ((D+n)/(n+1)),Real.exp
      (partialLSE_full (fun i : Fin D => input (row*D+i.val)) block.val
        (h_tail_of_lt_cdiv block) hasScale scale)) : WithBot ℝ)=
      formula input row D hasScale scale := by
  rw [aggregate_partialLSE_eq_fullRowLSE _ n hasScale scale positive]
  simp [formula,Nat.ne_of_gt positive,fullRowLSE,LSE]

#axiomsClean block_real
#axiomsClean aggregate_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogsumexpFwd.RealModel
