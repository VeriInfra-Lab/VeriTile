import bench.tritonbench_g.triton_attention.Proofs.RealBackwardInner
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

structure BackwardSeparate (p : BackwardParameters) : Prop where
  dq_dk : p.dq≠p.dk
  dq_dv : p.dq≠p.dv
  dv_dk : p.dv≠p.dk
  input : ∀ r∈[p.q,p.k,p.v,p.gradient,p.maximum,p.delta],r≠p.dq ∧ r≠p.dk ∧ r≠p.dv

theorem BackwardSeparate.noDQ (p : BackwardParameters) (sep : BackwardSeparate p) : NoDQInputs p := by
  constructor <;> intro r hr <;> exact (sep.input r (by simp only [List.mem_cons,List.mem_singleton] at hr ⊢; tauto)).1

theorem outer_readonly (p : BackwardParameters) (s : State R) (mem : Source.Memory R)
    (b : Nat) (r : RegionName) (off : Nat) (hr : r≠p.dq ∧ r≠p.dk ∧ r≠p.dv) :
    backwardOuterAdvance model p s b mem r off=mem r off := by
  unfold backwardOuterAdvance
  rw [memoryMaskedFold_frame _ _ _ _ _ _ _ _ (by intro i hi hm;exact Or.inl hr.2.1)]
  rw [memoryMaskedFold_frame _ _ _ _ _ _ _ _ (by intro i hi hm;exact Or.inl hr.2.2)]
  exact backwardInner_readonly model p s mem b _ r off hr.1

theorem outer_reads (p : BackwardParameters) (s : State R) (mem : Source.Memory R)
    (μ : RegionName → Nat → ℝ) (h : BackwardReads p mem μ) (sep : BackwardSeparate p) (b : Nat) :
    BackwardReads p (backwardOuterAdvance model p s b mem) μ := by
  constructor
  · intro r hr off hoff
    rw [outer_readonly p s mem b r off (sep.input r (by simp only [List.mem_cons,List.mem_singleton] at hr ⊢; tauto))]
    exact h.1 r hr off hoff
  · intro r hr off hoff
    rw [outer_readonly p s mem b r off (sep.input r (by simp only [List.mem_cons,List.mem_singleton] at hr ⊢; tauto))]
    exact h.2 r hr off hoff

theorem outer_dq (p : BackwardParameters) (s : State R) (mem : Source.Memory R)
    (μ : RegionName → Nat → ℝ) (h : BackwardReads p mem μ) (sep : BackwardSeparate p)
    (keyBlock : Nat) (hk : keyBlock<p.blocks)
    (hb : ∀ b<p.blocks,∀ j : Fin p.blockM,BackwardRowBound p s b j)
    (dq : Nat → BLane p → ℝ)
    (hq : ∀ b, b<p.blocks → ∀ i,mem p.dq (backwardOffset p s b i)=.mk .fp32 (some (dq b i)))
    (b : Nat) (hbn : b<p.blocks) (i : BLane p) :
    backwardOuterAdvance model p s keyBlock mem p.dq (backwardOffset p s b i)=
      .mk .fp32 (some (dq b i+if keyBlock≤b then queryContribution p s μ keyBlock b i else 0)) := by
  unfold backwardOuterAdvance
  rw [memoryMaskedFold_frame _ _ _ _ _ _ _ _ (by intro j hj hm;exact Or.inl sep.dq_dk)]
  rw [memoryMaskedFold_frame _ _ _ _ _ _ _ _ (by intro j hj hm;exact Or.inl sep.dq_dv)]
  by_cases he : keyBlock≤b
  · rw [inner_dq p s mem μ h sep.noDQ keyBlock hk hb dq hq _ (by omega) b (by omega) i,if_pos he]
  · rw [backwardInner_otherBlock model p s mem keyBlock _ b i (Or.inl (by omega)),hq b hbn i,if_neg he,add_zero]

theorem pointer_inBounds (p : BackwardParameters) (s : State R) (r : RegionName) (b : Nat)
    (hb : ∀ j : Fin p.blockM,backwardHeadRow p s+b*p.blockM+j.val<p.rows) (i : BLane p) :
    (backwardPointer p s r b).inBounds (TileShape.indexToList _ i) [0,1]=Bool.true := by
  simp [backwardPointer,TileShape.blockPtr_inBounds_2d_offsets_index,hb i.1,i.2.1.isLt]

theorem outer_dk (p : BackwardParameters) (s : State R) (mem : Source.Memory R)
    (μ : RegionName → Nat → ℝ) (h : BackwardReads p mem μ) (sep : BackwardSeparate p)
    (keyBlock : Nat) (hk : keyBlock<p.blocks)
    (hb : ∀ b<p.blocks,∀ j : Fin p.blockM,BackwardRowBound p s b j)
    (i : BLane p) :
    backwardOuterAdvance model p s keyBlock mem p.dk (backwardOffset p s keyBlock i)=
      .mk p.dtype.toTileDType (ofFloat p.dtype (some
        (∑ t ∈ Finset.range (p.blocks-keyBlock),keyContribution p s μ keyBlock (keyBlock+t) i))) := by
  unfold backwardOuterAdvance
  simp only [pointer_inBounds p s p.dk keyBlock (fun j => (hb keyBlock hk j).1),ite_true]
  rw [memoryFold_readback _ _ _ _ i (backwardOffset_injective p s keyBlock)]
  rw [(inner_sums p s mem μ h sep.noDQ keyBlock hk hb _ (by omega)).1 i]
  rfl

theorem outer_dv (p : BackwardParameters) (s : State R) (mem : Source.Memory R)
    (μ : RegionName → Nat → ℝ) (h : BackwardReads p mem μ) (sep : BackwardSeparate p)
    (keyBlock : Nat) (hk : keyBlock<p.blocks)
    (hb : ∀ b<p.blocks,∀ j : Fin p.blockM,BackwardRowBound p s b j)
    (i : BLane p) :
    backwardOuterAdvance model p s keyBlock mem p.dv (backwardOffset p s keyBlock i)=
      .mk p.dtype.toTileDType (ofFloat p.dtype (some
        (∑ t ∈ Finset.range (p.blocks-keyBlock),valueContribution p s μ keyBlock (keyBlock+t) i))) := by
  unfold backwardOuterAdvance
  rw [memoryMaskedFold_frame _ _ _ _ _ _ _ _ (by intro j hj hm;exact Or.inl sep.dv_dk)]
  simp only [pointer_inBounds p s p.dv keyBlock (fun j => (hb keyBlock hk j).1),ite_true]
  rw [memoryFold_readback _ _ _ _ i (backwardOffset_injective p s keyBlock)]
  rw [(inner_sums p s mem μ h sep.noDQ keyBlock hk hb _ (by omega)).2 i]
  rfl

theorem outer_otherKey (p : BackwardParameters) (s : State R) (mem : Source.Memory R)
    (sep : BackwardSeparate p) (keyBlock b : Nat) (he : b≠keyBlock) (r : RegionName)
    (hr : r=p.dk ∨ r=p.dv) (i : BLane p) :
    backwardOuterAdvance model p s keyBlock mem r (backwardOffset p s b i)=
      mem r (backwardOffset p s b i) := by
  unfold backwardOuterAdvance
  rw [memoryMaskedFold_frame]
  · rw [memoryMaskedFold_frame]
    · apply backwardInner_readonly
      rcases hr with rfl|rfl
      · exact Ne.symm sep.dq_dk
      · exact Ne.symm sep.dq_dv
    · intro j hj hm
      exact Or.inr (fun heq=>he (backwardOffset_inj p s b keyBlock i j heq).1)
  · intro j hj hm
    exact Or.inr (fun heq=>he (backwardOffset_inj p s b keyBlock i j heq).1)

#axiomsClean outer_dq
#axiomsClean outer_dk
#axiomsClean outer_dv
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
