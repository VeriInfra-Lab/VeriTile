import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealBackwardContributions
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.BackwardMemory
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def NoDQInputs (p : BackwardParameters) : Prop :=
  (∀ r∈[p.q,p.k,p.v,p.gradient],r≠p.dq) ∧ (∀ r∈[p.maximum,p.delta],r≠p.dq)

theorem inner_reads (p : BackwardParameters) (s : State R) (mem : Source.Memory R)
    (μ : RegionName → Nat → ℝ) (h : BackwardReads p mem μ) (hd : NoDQInputs p) (keyBlock n : Nat) :
    BackwardReads p (backwardInnerRounds model p s mem keyBlock n).mem μ := by
  constructor
  · intro r hr off hoff
    rw [backwardInner_readonly model p s mem keyBlock n r off (hd.1 r hr)]
    exact h.1 r hr off hoff
  · intro r hr off hoff
    rw [backwardInner_readonly model p s mem keyBlock n r off (hd.2 r hr)]
    exact h.2 r hr off hoff

theorem inner_sums (p : BackwardParameters) (s : State R) (mem : Source.Memory R)
    (μ : RegionName → Nat → ℝ) (h : BackwardReads p mem μ) (hd : NoDQInputs p)
    (keyBlock : Nat) (hk : keyBlock<p.blocks)
    (hb : ∀ b<p.blocks,∀ j : Fin p.blockM,BackwardRowBound p s b j)
    (n : Nat) (hn : keyBlock+n≤p.blocks) :
    (∀ i : BLane p,(backwardInnerRounds model p s mem keyBlock n).dk i=
      some (∑ t ∈ Finset.range n,keyContribution p s μ keyBlock (keyBlock+t) i)) ∧
    (∀ i : BLane p,(backwardInnerRounds model p s mem keyBlock n).dv i=
      some (∑ t ∈ Finset.range n,valueContribution p s μ keyBlock (keyBlock+t) i)) := by
  have hkey : ∀ i : BLane p,toFloat p.dtype (backwardRead p s mem p.k keyBlock i)=
      some (bAt p s μ p.k (keyBlock*p.blockM+i.1.val) i.2.1.val) :=
    fun i=>backwardRead_real p s mem μ h p.k (by simp) keyBlock i (hb keyBlock hk i.1).1
  have hval : ∀ i : BLane p,toFloat p.dtype (backwardRead p s mem p.v keyBlock i)=
      some (bAt p s μ p.v (keyBlock*p.blockM+i.1.val) i.2.1.val) :=
    fun i=>backwardRead_real p s mem μ h p.v (by simp) keyBlock i (hb keyBlock hk i.1).1
  induction n with
  | zero=>constructor <;> intro i <;> simp [backwardInnerRounds,backwardInnerInitial,model,algebra]
  | succ n ih=>
    have prev:=ih (by omega)
    have hr:=inner_reads p s mem μ h hd keyBlock n
    constructor
    · intro i
      change WithBot.realAdd ((backwardInnerRounds model p s mem keyBlock n).dk i)
        (backwardKeyDot model p s (backwardInnerRounds model p s mem keyBlock n).mem
          (backwardRead p s mem p.k keyBlock) (backwardRead p s mem p.v keyBlock) keyBlock (keyBlock+n) i)=_
      rw [prev.1 i,keyContribution_real p s _ μ hr _ _ keyBlock (keyBlock+n) hkey hval (hb _ (by omega)) i]
      simp [Finset.sum_range_succ,WithBot.realAdd]
    · intro i
      change WithBot.realAdd ((backwardInnerRounds model p s mem keyBlock n).dv i)
        (backwardValueDot model p s (backwardInnerRounds model p s mem keyBlock n).mem
          (backwardRead p s mem p.k keyBlock) keyBlock (keyBlock+n) i)=_
      rw [prev.2 i,valueContribution_real p s _ μ hr _ keyBlock (keyBlock+n) hkey (hb _ (by omega)) i]
      simp [Finset.sum_range_succ,WithBot.realAdd]

theorem inner_dq (p : BackwardParameters) (s : State R) (mem : Source.Memory R)
    (μ : RegionName → Nat → ℝ) (h : BackwardReads p mem μ) (hd : NoDQInputs p)
    (keyBlock : Nat) (hk : keyBlock<p.blocks)
    (hb : ∀ b<p.blocks,∀ j : Fin p.blockM,BackwardRowBound p s b j)
    (dq : Nat → BLane p → ℝ)
    (hq : ∀ b, b<p.blocks → ∀ i,mem p.dq (backwardOffset p s b i)=.mk .fp32 (some (dq b i)))
    (n : Nat) (hn : keyBlock+n≤p.blocks) (b : Nat) (hbn : keyBlock≤b ∧ b<keyBlock+n) (i : BLane p) :
    (backwardInnerRounds model p s mem keyBlock n).mem p.dq (backwardOffset p s b i)=
      .mk .fp32 (some (dq b i+queryContribution p s μ keyBlock b i)) := by
  have hkey : ∀ i : BLane p,toFloat p.dtype (backwardRead p s mem p.k keyBlock i)=
      some (bAt p s μ p.k (keyBlock*p.blockM+i.1.val) i.2.1.val) :=
    fun i=>backwardRead_real p s mem μ h p.k (by simp) keyBlock i (hb keyBlock hk i.1).1
  have hval : ∀ i : BLane p,toFloat p.dtype (backwardRead p s mem p.v keyBlock i)=
      some (bAt p s μ p.v (keyBlock*p.blockM+i.1.val) i.2.1.val) :=
    fun i=>backwardRead_real p s mem μ h p.v (by simp) keyBlock i (hb keyBlock hk i.1).1
  induction n with
  | zero=>omega
  | succ n ih=>
    change ((TileShape.allIndices [p.blockM,p.head]).foldl (fun m j=>writeMemory m p.dq
      (backwardOffset p s (keyBlock+n) j) _) _) p.dq (backwardOffset p s b i)=_
    by_cases he : b=keyBlock+n
    · subst b
      rw [memoryFold_readback _ _ _ _ i (backwardOffset_injective p s (keyBlock+n))]
      simp only [backwardQueryCell,backwardQueryValue]
      rw [backwardInner_otherBlock model p s mem keyBlock n (keyBlock+n) i (Or.inr (by omega)),hq _ (by omega)]
      rw [queryContribution_real p s _ μ (inner_reads p s mem μ h hd keyBlock n) _ _ keyBlock (keyBlock+n)
        hkey hval i (hb _ (by omega) i.1)]
      rfl
    · rw [memoryFold_frame]
      · exact ih (by omega) (by omega)
      · intro j _
        right
        intro heq
        exact he (backwardOffset_inj p s b (keyBlock+n) i j heq).1

#axiomsClean inner_sums
#axiomsClean inner_dq
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
