import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealPreprocess
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPBackwardFrame
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

/-- Current input and saved-scalar loads. DQ/DK/DV are deliberately excluded. -/
def BackwardReads (p : BackwardParameters) (mem : Source.Memory R) (μ : RegionName → Nat → ℝ) : Prop :=
  (∀ r ∈ [p.q,p.k,p.v,p.gradient],∀ off,off < p.rows*p.head → toFloat p.dtype ((mem r off).read p.dtype.toTileDType)=some (μ r off)) ∧
  (∀ r ∈ [p.maximum,p.delta],∀ off,off < p.rows → (mem r off).read .fp32=some (μ r off))
def BackwardRowBound (p : BackwardParameters) (s : State R) (b : Nat) (j : Fin p.blockM) : Prop :=
  backwardHeadRow p s+b*p.blockM+j.val<p.rows ∧ backwardRowOffset p s b (j,())<p.rows

def bAt (p : BackwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (r : RegionName) (row d : Nat) : ℝ := μ r ((backwardHeadRow p s+row)*p.head+d)
def bRow (p : BackwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (r : RegionName) (row : Nat) : ℝ := μ r (s.pids 0*(p.blockM*p.blocks)+row)

def backwardProbability (dim : Nat) (scale : ℝ) (q k : Nat → Nat → ℝ) (m : Nat → ℝ) (i j : Nat) : ℝ :=
  if j ≤ i then Real.exp (Mathematical.score dim scale q k i j-m i) else 0
def backwardScoreGradient (dim : Nat) (scale : ℝ) (q k v grad : Nat → Nat → ℝ)
    (m delta : Nat → ℝ) (i j : Nat) : ℝ :=
  backwardProbability dim scale q k m i j*((∑ d ∈ Finset.range dim,grad i d*v j d)-delta i)*scale

theorem backwardRead_real (p : BackwardParameters) (s : State R)
    (mem : Source.Memory R) (μ : RegionName → Nat → ℝ) (h : BackwardReads p mem μ)
    (r : RegionName) (hr : r ∈ [p.q,p.k,p.v,p.gradient]) (c : Nat) (i : BLane p)
    (hb : backwardHeadRow p s+c*p.blockM+i.1.val<p.rows) :
    toFloat p.dtype (backwardRead p s mem r c i)=some (bAt p s μ r (c*p.blockM+i.1.val) i.2.1.val) := by
  simp only [backwardRead,backwardPointer,TileShape.blockPtr_inBounds_2d_offsets_index,Nat.zero_add,
    show backwardHeadRow p s+c*p.blockM+i.1.val<p.rows ∧ i.2.1.val<p.head from ⟨hb,i.2.1.isLt⟩]
  simpa [backwardOffset,bAt,Nat.add_assoc] using h.1 r hr (backwardOffset p s c i) (by
    dsimp only [backwardOffset]
    have hd := i.2.1.isLt
    have hh := Nat.mul_le_mul_right p.head (Nat.succ_le_of_lt hb)
    nlinarith)

theorem backwardQK_real (p : BackwardParameters) (s : State R)
    (mem : Source.Memory R) (μ : RegionName → Nat → ℝ) (h : BackwardReads p mem μ)
    (key : Values R p.dtype.toTileDType [p.blockM,p.head]) (keyBlock queryBlock : Nat)
    (hk : ∀ i : BLane p,toFloat p.dtype (key i)=some (bAt p s μ p.k (keyBlock*p.blockM+i.1.val) i.2.1.val))
    (i : BScore p) (hb : backwardHeadRow p s+queryBlock*p.blockM+i.1.val<p.rows) :
    backwardQK model p s mem key queryBlock i=some
      (∑ d ∈ Finset.range p.head,bAt p s μ p.q (queryBlock*p.blockM+i.1.val) d*bAt p s μ p.k (keyBlock*p.blockM+i.2.1.val) d) := by
  change sum p.head (fun d=>WithBot.realMul
    (toFloat p.dtype (backwardRead p s mem p.q queryBlock (i.1,d,())))
    (toFloat p.dtype (key (i.2.1,d,()))))=_
  have he : (fun d : Fin p.head=>WithBot.realMul
      (toFloat p.dtype (backwardRead p s mem p.q queryBlock (i.1,d,())))
      (toFloat p.dtype (key (i.2.1,d,()))))=
      (fun d=>some (bAt p s μ p.q (queryBlock*p.blockM+i.1.val) d.val*bAt p s μ p.k (keyBlock*p.blockM+i.2.1.val) d.val)) := by
    funext d
    rw [backwardRead_real p s mem μ h p.q (by simp) queryBlock (i.1,d,()) hb,hk (i.2.1,d,())]
    rfl
  rw [he,sum_some]
  congr 1
  exact Fin.sum_univ_eq_sum_range (fun d : Nat=>bAt p s μ p.q (queryBlock*p.blockM+i.1.val) d*bAt p s μ p.k (keyBlock*p.blockM+i.2.1.val) d) p.head

theorem backwardWeights_real (p : BackwardParameters) (s : State R)
    (mem : Source.Memory R) (μ : RegionName → Nat → ℝ) (h : BackwardReads p mem μ)
    (key : Values R p.dtype.toTileDType [p.blockM,p.head]) (keyBlock queryBlock : Nat)
    (hk : ∀ i : BLane p,toFloat p.dtype (key i)=some (bAt p s μ p.k (keyBlock*p.blockM+i.1.val) i.2.1.val))
    (i : BScore p) (hb : BackwardRowBound p s queryBlock i.1) :
    backwardWeights model p s mem key keyBlock queryBlock i=some
      (backwardProbability p.head p.scale (bAt p s μ p.q) (bAt p s μ p.k) (bRow p s μ p.maximum)
        (queryBlock*p.blockM+i.1.val) (keyBlock*p.blockM+i.2.1.val)) := by
  have hm:=h.2 p.maximum (by simp) (backwardRowOffset p s queryBlock (i.1,())) hb.2
  simp only [backwardWeights,hm,backwardMaskedQK]
  by_cases hv : keyBlock*p.blockM+i.2.1.val ≤ queryBlock*p.blockM+i.1.val
  · rw [if_pos hv,backwardQK_real p s mem μ h key keyBlock queryBlock hk i hb.1]
    simp [model,algebra,backwardProbability,Mathematical.score,hv,exp,WithBot.realMul,WithBot.realSub,bRow,backwardRowOffset,Nat.add_assoc]
  · simp [model,algebra,backwardProbability,hv,exp,WithBot.realMul,WithBot.realSub]

theorem backwardDPDot_real (p : BackwardParameters) (s : State R)
    (mem : Source.Memory R) (μ : RegionName → Nat → ℝ) (h : BackwardReads p mem μ)
    (value : Values R p.dtype.toTileDType [p.blockM,p.head]) (keyBlock queryBlock : Nat)
    (hv : ∀ i : BLane p,toFloat p.dtype (value i)=some (bAt p s μ p.v (keyBlock*p.blockM+i.1.val) i.2.1.val))
    (i : BScore p) (hb : backwardHeadRow p s+queryBlock*p.blockM+i.1.val<p.rows) :
    backwardDPDot model p s mem value queryBlock i=some
      (∑ d ∈ Finset.range p.head,bAt p s μ p.gradient (queryBlock*p.blockM+i.1.val) d*bAt p s μ p.v (keyBlock*p.blockM+i.2.1.val) d) := by
  change sum p.head (fun d=>WithBot.realMul
    (toFloat p.dtype (backwardRead p s mem p.gradient queryBlock (i.1,d,())))
    (toFloat p.dtype (value (i.2.1,d,()))))=_
  have he : (fun d : Fin p.head=>WithBot.realMul
      (toFloat p.dtype (backwardRead p s mem p.gradient queryBlock (i.1,d,())))
      (toFloat p.dtype (value (i.2.1,d,()))))=
      (fun d=>some (bAt p s μ p.gradient (queryBlock*p.blockM+i.1.val) d.val*bAt p s μ p.v (keyBlock*p.blockM+i.2.1.val) d.val)) := by
    funext d
    rw [backwardRead_real p s mem μ h p.gradient (by simp) queryBlock (i.1,d,()) hb,hv (i.2.1,d,())]
    rfl
  rw [he,sum_some]
  congr 1
  exact Fin.sum_univ_eq_sum_range (fun d : Nat=>bAt p s μ p.gradient (queryBlock*p.blockM+i.1.val) d*bAt p s μ p.v (keyBlock*p.blockM+i.2.1.val) d) p.head

theorem backwardDS_real (p : BackwardParameters) (s : State R)
    (mem : Source.Memory R) (μ : RegionName → Nat → ℝ) (h : BackwardReads p mem μ)
    (key value : Values R p.dtype.toTileDType [p.blockM,p.head]) (keyBlock queryBlock : Nat)
    (hk : ∀ i : BLane p,toFloat p.dtype (key i)=some (bAt p s μ p.k (keyBlock*p.blockM+i.1.val) i.2.1.val))
    (hv : ∀ i : BLane p,toFloat p.dtype (value i)=some (bAt p s μ p.v (keyBlock*p.blockM+i.1.val) i.2.1.val))
    (i : BScore p) (hb : BackwardRowBound p s queryBlock i.1) :
    backwardDS model p s mem key value keyBlock queryBlock i=some
      (backwardScoreGradient p.head p.scale (bAt p s μ p.q) (bAt p s μ p.k)
        (bAt p s μ p.v) (bAt p s μ p.gradient) (bRow p s μ p.maximum) (bRow p s μ p.delta)
        (queryBlock*p.blockM+i.1.val) (keyBlock*p.blockM+i.2.1.val)) := by
  have hd:=h.2 p.delta (by simp) (backwardRowOffset p s queryBlock (i.1,())) hb.2
  simp only [backwardDS,backwardDP,backwardDPInitial,hd]
  rw [backwardWeights_real p s mem μ h key keyBlock queryBlock hk i hb,
    backwardDPDot_real p s mem μ h value keyBlock queryBlock hv i hb.1]
  simp [model,algebra,backwardScoreGradient,WithBot.realMul,WithBot.realSub,WithBot.realAdd,bRow,backwardRowOffset,Nat.add_assoc,sub_eq_add_neg,add_comm]

#axiomsClean backwardWeights_real
#axiomsClean backwardDS_real
#specNonCircular backwardScoreGradient avoiding [FP.Structural.exec,backwardRounds,backwardFinalMemory]
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
