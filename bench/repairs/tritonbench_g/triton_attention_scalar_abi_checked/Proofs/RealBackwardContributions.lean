import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealBackwardValues
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def bP (p : BackwardParameters) (s : State R) (μ : RegionName → Nat → ℝ) (i j : Nat) : ℝ :=
  backwardProbability p.head p.scale (bAt p s μ p.q) (bAt p s μ p.k) (bRow p s μ p.maximum) i j
def bDS (p : BackwardParameters) (s : State R) (μ : RegionName → Nat → ℝ) (i j : Nat) : ℝ :=
  backwardScoreGradient p.head p.scale (bAt p s μ p.q) (bAt p s μ p.k)
    (bAt p s μ p.v) (bAt p s μ p.gradient) (bRow p s μ p.maximum) (bRow p s μ p.delta) i j

def valueContribution (p : BackwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (keyBlock queryBlock : Nat) (i : BLane p) : ℝ :=
  ∑ j ∈ Finset.range p.blockM,bP p s μ (queryBlock*p.blockM+j) (keyBlock*p.blockM+i.1.val)*
    bAt p s μ p.gradient (queryBlock*p.blockM+j) i.2.1.val

def keyContribution (p : BackwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (keyBlock queryBlock : Nat) (i : BLane p) : ℝ :=
  ∑ j ∈ Finset.range p.blockM,bDS p s μ (queryBlock*p.blockM+j) (keyBlock*p.blockM+i.1.val)*
    bAt p s μ p.q (queryBlock*p.blockM+j) i.2.1.val

def queryContribution (p : BackwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (keyBlock queryBlock : Nat) (i : BLane p) : ℝ :=
  ∑ j ∈ Finset.range p.blockM,bDS p s μ (queryBlock*p.blockM+i.1.val) (keyBlock*p.blockM+j)*
    bAt p s μ p.k (keyBlock*p.blockM+j) i.2.1.val

theorem valueContribution_real (p : BackwardParameters) (s : State R)
    (mem : Source.Memory R) (μ : RegionName → Nat → ℝ) (h : BackwardReads p mem μ)
    (key : Values R p.dtype.toTileDType [p.blockM,p.head]) (keyBlock queryBlock : Nat)
    (hk : ∀ i : BLane p,toFloat p.dtype (key i)=some (bAt p s μ p.k (keyBlock*p.blockM+i.1.val) i.2.1.val))
    (hb : ∀ j : Fin p.blockM,BackwardRowBound p s queryBlock j)
    (i : BLane p) :
    backwardValueDot model p s mem key keyBlock queryBlock i=some (valueContribution p s μ keyBlock queryBlock i) := by
  change sum p.blockM (fun j=>WithBot.realMul
    (toFloat p.dtype (ofFloat p.dtype (backwardWeights model p s mem key keyBlock queryBlock (j,i.1,()))))
    (toFloat p.dtype (backwardRead p s mem p.gradient queryBlock (j,i.2.1,()))))=_
  simp only [toFloat_ofFloat_real]
  have he : (fun j : Fin p.blockM=>WithBot.realMul
      (backwardWeights model p s mem key keyBlock queryBlock (j,i.1,()))
      (toFloat p.dtype (backwardRead p s mem p.gradient queryBlock (j,i.2.1,()))))=
      (fun j=>some (bP p s μ (queryBlock*p.blockM+j.val) (keyBlock*p.blockM+i.1.val)*
        bAt p s μ p.gradient (queryBlock*p.blockM+j.val) i.2.1.val)) := by
    funext j
    rw [backwardWeights_real p s mem μ h key keyBlock queryBlock hk (j,i.1,()) (hb j),
      backwardRead_real p s mem μ h p.gradient (by simp) queryBlock (j,i.2.1,()) (hb j).1]
    rfl
  rw [he,sum_some]
  congr 1
  exact Fin.sum_univ_eq_sum_range (fun j : Nat=>bP p s μ (queryBlock*p.blockM+j) (keyBlock*p.blockM+i.1.val)*
    bAt p s μ p.gradient (queryBlock*p.blockM+j) i.2.1.val) p.blockM

theorem keyContribution_real (p : BackwardParameters) (s : State R)
    (mem : Source.Memory R) (μ : RegionName → Nat → ℝ) (h : BackwardReads p mem μ)
    (key value : Values R p.dtype.toTileDType [p.blockM,p.head]) (keyBlock queryBlock : Nat)
    (hk : ∀ i : BLane p,toFloat p.dtype (key i)=some (bAt p s μ p.k (keyBlock*p.blockM+i.1.val) i.2.1.val))
    (hv : ∀ i : BLane p,toFloat p.dtype (value i)=some (bAt p s μ p.v (keyBlock*p.blockM+i.1.val) i.2.1.val))
    (hb : ∀ j : Fin p.blockM,BackwardRowBound p s queryBlock j)
    (i : BLane p) :
    backwardKeyDot model p s mem key value keyBlock queryBlock i=some (keyContribution p s μ keyBlock queryBlock i) := by
  change sum p.blockM (fun j=>WithBot.realMul
    (toFloat p.dtype (ofFloat p.dtype (backwardDS model p s mem key value keyBlock queryBlock (j,i.1,()))))
    (toFloat p.dtype (backwardRead p s mem p.q queryBlock (j,i.2.1,()))))=_
  simp only [toFloat_ofFloat_real]
  have he : (fun j : Fin p.blockM=>WithBot.realMul
      (backwardDS model p s mem key value keyBlock queryBlock (j,i.1,()))
      (toFloat p.dtype (backwardRead p s mem p.q queryBlock (j,i.2.1,()))))=
      (fun j=>some (bDS p s μ (queryBlock*p.blockM+j.val) (keyBlock*p.blockM+i.1.val)*
        bAt p s μ p.q (queryBlock*p.blockM+j.val) i.2.1.val)) := by
    funext j
    rw [backwardDS_real p s mem μ h key value keyBlock queryBlock hk hv (j,i.1,()) (hb j),
      backwardRead_real p s mem μ h p.q (by simp) queryBlock (j,i.2.1,()) (hb j).1]
    rfl
  rw [he,sum_some]
  congr 1
  exact Fin.sum_univ_eq_sum_range (fun j : Nat=>bDS p s μ (queryBlock*p.blockM+j) (keyBlock*p.blockM+i.1.val)*
    bAt p s μ p.q (queryBlock*p.blockM+j) i.2.1.val) p.blockM

theorem queryContribution_real (p : BackwardParameters) (s : State R)
    (mem : Source.Memory R) (μ : RegionName → Nat → ℝ) (h : BackwardReads p mem μ)
    (key value : Values R p.dtype.toTileDType [p.blockM,p.head]) (keyBlock queryBlock : Nat)
    (hk : ∀ i : BLane p,toFloat p.dtype (key i)=some (bAt p s μ p.k (keyBlock*p.blockM+i.1.val) i.2.1.val))
    (hv : ∀ i : BLane p,toFloat p.dtype (value i)=some (bAt p s μ p.v (keyBlock*p.blockM+i.1.val) i.2.1.val))
    (i : BLane p) (hb : BackwardRowBound p s queryBlock i.1) :
    backwardQueryDot model p s mem key value keyBlock queryBlock i=some (queryContribution p s μ keyBlock queryBlock i) := by
  change sum p.blockM (fun j=>WithBot.realMul
    (toFloat p.dtype (ofFloat p.dtype (backwardDS model p s mem key value keyBlock queryBlock (i.1,j,()))))
    (toFloat p.dtype (key (j,i.2.1,()))))=_
  simp only [toFloat_ofFloat_real]
  have he : (fun j : Fin p.blockM=>WithBot.realMul
      (backwardDS model p s mem key value keyBlock queryBlock (i.1,j,()))
      (toFloat p.dtype (key (j,i.2.1,()))))=
      (fun j=>some (bDS p s μ (queryBlock*p.blockM+i.1.val) (keyBlock*p.blockM+j.val)*
        bAt p s μ p.k (keyBlock*p.blockM+j.val) i.2.1.val)) := by
    funext j
    rw [backwardDS_real p s mem μ h key value keyBlock queryBlock hk hv (i.1,j,()) hb,hk (j,i.2.1,())]
    rfl
  rw [he,sum_some]
  congr 1
  exact Fin.sum_univ_eq_sum_range (fun j : Nat=>bDS p s μ (queryBlock*p.blockM+i.1.val) (keyBlock*p.blockM+j)*
    bAt p s μ p.k (keyBlock*p.blockM+j) i.2.1.val) p.blockM

#axiomsClean valueContribution_real
#axiomsClean keyContribution_real
#axiomsClean queryContribution_real
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
