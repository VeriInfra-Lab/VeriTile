import bench.tritonbench_g.triton_attention.Proofs.NativeBackwardValues
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 10000000

theorem lower_seeded_dot (one : Bool) (seed : Stage) (w : Width) (k : Nat) (a b : Nat → Graph) :
    lowerScalar one .backward (.add seed (.dot w k a b))=dot w k a b Bool.true (lowerScalar one .backward seed) := by
  cases seed <;> rfl

theorem backwardValueDot_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr) (mem : Source.Memory Expr)
    (x : Target.BackwardInputs) (head kb qb : Nat) (reads : BackwardReady c scale s mem x head) (hqb : qb<c.blocks)
    (key : Values Expr .fp32 [128,c.dim])
    (hk : ∀ i : BLane (backwardParameters c scale),key i=fun _=>Stage.ready (x.k head (kb*128+i.1.val) i.2.1.val))
    (i : BLane (backwardParameters c scale)) (u : Nat) :
    backwardValueDot (modelScalar one .backward) (backwardParameters c scale) s mem key kb qb i u=
      Stage.dot (TritonAttention.Whole001.Native.width c.dim) 128
        (Target.pad 128 (fun j=>Target.half (Target.backwardProbability c one scale x head (qb*128+j) (kb*128+i.1.val))))
        (Target.pad 128 (fun j=>x.gradient head (qb*128+j) i.2.1.val)) := by
  change Stage.dot (TritonAttention.Whole001.Native.width c.dim) 128
    (fun d=>if hd : d<128 then Target.half (lowerScalar one .backward
      (backwardWeights (modelScalar one .backward) (backwardParameters c scale) s mem key kb qb (⟨d,hd⟩,i.1,()) (Reduction.owner d).val)) else .word 0)
    (fun d=>if hd : d<128 then lowerScalar one .backward
      (backwardRead (backwardParameters c scale) s mem "NewDO" qb (⟨d,hd⟩,i.2.1,()) (Reduction.owner i.2.1.val).val) else .word 0)=_
  congr 1
  · funext j
    by_cases hj : j<128
    · simp only [Target.pad,dif_pos hj,if_pos hj]
      rw [backwardWeights_native c scale one s mem x head kb qb reads hqb key hk]
    · simp only [Target.pad,dif_neg hj,if_neg hj]
  · funext j
    by_cases hj : j<128
    · simp only [Target.pad,dif_pos hj,if_pos hj]
      rw [reads.grad qb hqb]
      rfl
    · simp only [Target.pad,dif_neg hj,if_neg hj]

theorem backwardKeyDot_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr) (mem : Source.Memory Expr)
    (x : Target.BackwardInputs) (head kb qb : Nat) (reads : BackwardReady c scale s mem x head) (hqb : qb<c.blocks)
    (key value : Values Expr .fp32 [128,c.dim])
    (hk : ∀ i : BLane (backwardParameters c scale),key i=fun _=>Stage.ready (x.k head (kb*128+i.1.val) i.2.1.val))
    (hv : ∀ i : BLane (backwardParameters c scale),value i=fun _=>Stage.ready (x.v head (kb*128+i.1.val) i.2.1.val))
    (i : BLane (backwardParameters c scale)) (u : Nat) :
    backwardKeyDot (modelScalar one .backward) (backwardParameters c scale) s mem key value kb qb i u=
      Stage.dot (TritonAttention.Whole001.Native.width c.dim) 128
        (Target.pad 128 (fun j=>Target.half (Target.backwardDS c one scale x head (qb*128+j) (kb*128+i.1.val))))
        (Target.pad 128 (fun j=>x.q head (qb*128+j) i.2.1.val)) := by
  change Stage.dot (TritonAttention.Whole001.Native.width c.dim) 128
    (fun d=>if hd : d<128 then Target.half (lowerScalar one .backward
      (backwardDS (modelScalar one .backward) (backwardParameters c scale) s mem key value kb qb (⟨d,hd⟩,i.1,()) (Reduction.owner d).val)) else .word 0)
    (fun d=>if hd : d<128 then lowerScalar one .backward
      (backwardRead (backwardParameters c scale) s mem "Q" qb (⟨d,hd⟩,i.2.1,()) (Reduction.owner i.2.1.val).val) else .word 0)=_
  congr 1
  · funext j
    by_cases hj : j<128
    · simp only [Target.pad,dif_pos hj,if_pos hj]
      rw [backwardDS_native c scale one s mem x head kb qb reads hqb key value hk hv]
    · simp only [Target.pad,dif_neg hj,if_neg hj]
  · funext j
    by_cases hj : j<128
    · simp only [Target.pad,dif_pos hj,if_pos hj]
      rw [reads.q qb hqb]
      rfl
    · simp only [Target.pad,dif_neg hj,if_neg hj]

theorem backwardQueryDot_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr) (mem : Source.Memory Expr)
    (x : Target.BackwardInputs) (head kb qb : Nat) (reads : BackwardReady c scale s mem x head) (hqb : qb<c.blocks)
    (key value : Values Expr .fp32 [128,c.dim])
    (hk : ∀ i : BLane (backwardParameters c scale),key i=fun _=>Stage.ready (x.k head (kb*128+i.1.val) i.2.1.val))
    (hv : ∀ i : BLane (backwardParameters c scale),value i=fun _=>Stage.ready (x.v head (kb*128+i.1.val) i.2.1.val))
    (i : BLane (backwardParameters c scale)) (u : Nat) :
    backwardQueryDot (modelScalar one .backward) (backwardParameters c scale) s mem key value kb qb i u=
      Stage.dot (TritonAttention.Whole001.Native.width c.dim) 128
        (Target.pad 128 (fun j=>Target.half (Target.backwardDS c one scale x head (qb*128+i.1.val) (kb*128+j))))
        (Target.pad 128 (fun j=>x.k head (kb*128+j) i.2.1.val)) := by
  change Stage.dot (TritonAttention.Whole001.Native.width c.dim) 128
    (fun d=>if hd : d<128 then Target.half (lowerScalar one .backward
      (backwardDS (modelScalar one .backward) (backwardParameters c scale) s mem key value kb qb (i.1,⟨d,hd⟩,()) (Reduction.owner d).val)) else .word 0)
    (fun d=>if hd : d<128 then lowerScalar one .backward
      (key (⟨d,hd⟩,i.2.1,()) (Reduction.owner i.2.1.val).val) else .word 0)=_
  congr 1
  · funext j
    by_cases hj : j<128
    · simp only [Target.pad,dif_pos hj,if_pos hj]
      rw [backwardDS_native c scale one s mem x head kb qb reads hqb key value hk hv]
    · simp only [Target.pad,dif_neg hj,if_neg hj]
  · funext j
    by_cases hj : j<128
    · simp only [Target.pad,dif_pos hj,if_pos hj]
      rw [hk]
      rfl
    · simp only [Target.pad,dif_neg hj,if_neg hj]

#axiomsClean backwardValueDot_native
#axiomsClean backwardKeyDot_native
#axiomsClean backwardQueryDot_native
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
