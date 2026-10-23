import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.NativeForwardValues
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 10000000

structure BackwardReady (c : Shape) (scale : ℝ) (s : State Expr) (mem : Source.Memory Expr)
    (x : Target.BackwardInputs) (head : Nat) : Prop where
  q : ∀ b,b<c.blocks → ∀ i : BLane (backwardParameters c scale),
    backwardRead (backwardParameters c scale) s mem "Q" b i=fun _=>Stage.ready (x.q head (b*128+i.1.val) i.2.1.val)
  k : ∀ b,b<c.blocks → ∀ i : BLane (backwardParameters c scale),
    backwardRead (backwardParameters c scale) s mem "K" b i=fun _=>Stage.ready (x.k head (b*128+i.1.val) i.2.1.val)
  v : ∀ b,b<c.blocks → ∀ i : BLane (backwardParameters c scale),
    backwardRead (backwardParameters c scale) s mem "V" b i=fun _=>Stage.ready (x.v head (b*128+i.1.val) i.2.1.val)
  grad : ∀ b,b<c.blocks → ∀ i : BLane (backwardParameters c scale),
    backwardRead (backwardParameters c scale) s mem "NewDO" b i=fun _=>Stage.ready (x.gradient head (b*128+i.1.val) i.2.1.val)
  maximum : ∀ b,b<c.blocks → ∀ i : Fin 128,
    (mem "M" (backwardRowOffset (backwardParameters c scale) s b (i,()))).read .fp32=
      fun _=>Stage.ready (x.maximum head (b*128+i.val))
  delta : ∀ b,b<c.blocks → ∀ i : Fin 128,
    (mem "Delta" (backwardRowOffset (backwardParameters c scale) s b (i,()))).read .fp32=
      fun _=>Stage.ready (x.delta head (b*128+i.val))

theorem backwardQK_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr) (mem : Source.Memory Expr)
    (x : Target.BackwardInputs) (head kb qb : Nat) (reads : BackwardReady c scale s mem x head) (hqb : qb<c.blocks)
    (key : Values Expr .fp32 [128,c.dim])
    (hk : ∀ i : BLane (backwardParameters c scale),key i=fun _=>Stage.ready (x.k head (kb*128+i.1.val) i.2.1.val))
    (i : BScore (backwardParameters c scale)) (u : Nat) :
    backwardQK (modelScalar one .backward) (backwardParameters c scale) s mem key qb i u=
      Stage.dot .n128 c.dim (Target.pad c.dim (x.q head (qb*128+i.1.val)))
        (Target.pad c.dim (x.k head (kb*128+i.2.1.val))) := by
  change Stage.dot .n128 c.dim
    (fun d=>if hd : d<c.dim then lowerScalar one .backward
      (backwardRead (backwardParameters c scale) s mem "Q" qb (i.1,⟨d,hd⟩,()) (Reduction.owner d).val) else .word 0)
    (fun d=>if hd : d<c.dim then lowerScalar one .backward
      (key (i.2.1,⟨d,hd⟩,()) (Reduction.owner i.2.1.val).val) else .word 0)=_
  congr 1
  · funext d
    by_cases hd : d<c.dim
    · simp only [Target.pad,dif_pos hd,if_pos hd]
      rw [reads.q qb hqb]
      rfl
    · simp only [Target.pad,dif_neg hd,if_neg hd]
  · funext d
    by_cases hd : d<c.dim
    · simp only [Target.pad,dif_pos hd,if_pos hd]
      rw [hk]
      rfl
    · simp only [Target.pad,dif_neg hd,if_neg hd]

theorem backwardWeights_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr) (mem : Source.Memory Expr)
    (x : Target.BackwardInputs) (head kb qb : Nat) (reads : BackwardReady c scale s mem x head) (hqb : qb<c.blocks)
    (key : Values Expr .fp32 [128,c.dim])
    (hk : ∀ i : BLane (backwardParameters c scale),key i=fun _=>Stage.ready (x.k head (kb*128+i.1.val) i.2.1.val))
    (i : BScore (backwardParameters c scale)) (u : Nat) :
    lowerScalar one .backward (backwardWeights (modelScalar one .backward) (backwardParameters c scale) s mem key kb qb i u)=
      Target.backwardProbability c one scale x head (qb*128+i.1.val) (kb*128+i.2.1.val) := by
  change Target.exp (lowerScalar one .backward (.sub
    (.mul (backwardMaskedQK (modelScalar one .backward) (backwardParameters c scale) s mem key kb qb i u) (literal scale))
    ((mem "M" (backwardRowOffset (backwardParameters c scale) s qb (i.1,()))).read .fp32 u)))=_
  rw [reads.maximum qb hqb,lower_scale_sub]
  unfold Target.backwardProbability
  congr 2
  unfold backwardMaskedQK
  simp only [show (backwardParameters c scale).blockM=128 from rfl]
  split_ifs
  · change lowerScalar one .backward (backwardQK (modelScalar one .backward) (backwardParameters c scale) s mem key qb i u)=_
    rw [backwardQK_native c scale one s mem x head kb qb reads hqb key hk i u]
    rfl
  · rfl

theorem backwardDP_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr) (mem : Source.Memory Expr)
    (x : Target.BackwardInputs) (head kb qb : Nat) (reads : BackwardReady c scale s mem x head) (hqb : qb<c.blocks)
    (value : Values Expr .fp32 [128,c.dim])
    (hv : ∀ i : BLane (backwardParameters c scale),value i=fun _=>Stage.ready (x.v head (kb*128+i.1.val) i.2.1.val))
    (i : BScore (backwardParameters c scale)) (u : Nat) :
    lowerScalar one .backward (backwardDP (modelScalar one .backward) (backwardParameters c scale) s mem value qb i u)=
      dot .n128 c.dim (Target.pad c.dim (x.gradient head (qb*128+i.1.val)))
        (Target.pad c.dim (x.v head (kb*128+i.2.1.val))) Bool.true (.neg (x.delta head (qb*128+i.1.val))) := by
  have hd : backwardDPDot (modelScalar one .backward) (backwardParameters c scale) s mem value qb i u=
      Stage.dot .n128 c.dim (Target.pad c.dim (x.gradient head (qb*128+i.1.val)))
        (Target.pad c.dim (x.v head (kb*128+i.2.1.val))) := by
    change Stage.dot .n128 c.dim
      (fun d=>if hd : d<c.dim then lowerScalar one .backward
        (backwardRead (backwardParameters c scale) s mem "NewDO" qb (i.1,⟨d,hd⟩,()) (Reduction.owner d).val) else .word 0)
      (fun d=>if hd : d<c.dim then lowerScalar one .backward
        (value (i.2.1,⟨d,hd⟩,()) (Reduction.owner i.2.1.val).val) else .word 0)=_
    congr 1
    · funext d
      by_cases hd : d<c.dim
      · simp only [Target.pad,dif_pos hd,if_pos hd]
        rw [reads.grad qb hqb]
        rfl
      · simp only [Target.pad,dif_neg hd,if_neg hd]
    · funext d
      by_cases hd : d<c.dim
      · simp only [Target.pad,dif_pos hd,if_pos hd]
        rw [hv]
        rfl
      · simp only [Target.pad,dif_neg hd,if_neg hd]
  change lowerScalar one .backward (.add
    (.sub (literal 0) ((mem "Delta" (backwardRowOffset (backwardParameters c scale) s qb (i.1,()))).read .fp32 u))
    (backwardDPDot (modelScalar one .backward) (backwardParameters c scale) s mem value qb i u))=_
  rw [hd,reads.delta qb hqb,show literal 0=Stage.zero from by simp [literal]]
  rfl

theorem backwardDS_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr) (mem : Source.Memory Expr)
    (x : Target.BackwardInputs) (head kb qb : Nat) (reads : BackwardReady c scale s mem x head) (hqb : qb<c.blocks)
    (key value : Values Expr .fp32 [128,c.dim])
    (hk : ∀ i : BLane (backwardParameters c scale),key i=fun _=>Stage.ready (x.k head (kb*128+i.1.val) i.2.1.val))
    (hv : ∀ i : BLane (backwardParameters c scale),value i=fun _=>Stage.ready (x.v head (kb*128+i.1.val) i.2.1.val))
    (i : BScore (backwardParameters c scale)) (u : Nat) :
    lowerScalar one .backward (backwardDS (modelScalar one .backward) (backwardParameters c scale) s mem key value kb qb i u)=
      Target.backwardDS c one scale x head (qb*128+i.1.val) (kb*128+i.2.1.val) := by
  change lowerScalar one .backward (.mul (.mul
    (backwardWeights (modelScalar one .backward) (backwardParameters c scale) s mem key kb qb i u)
    (backwardDP (modelScalar one .backward) (backwardParameters c scale) s mem value qb i u)) (literal scale))=_
  rw [lower_scale_mul]
  change Target.scaleMul one scale (.mul
    (lowerScalar one .backward (backwardWeights (modelScalar one .backward) (backwardParameters c scale) s mem key kb qb i u))
    (lowerScalar one .backward (backwardDP (modelScalar one .backward) (backwardParameters c scale) s mem value qb i u)))=_
  rw [backwardWeights_native c scale one s mem x head kb qb reads hqb key hk,
    backwardDP_native c scale one s mem x head kb qb reads hqb value hv]
  rfl

#axiomsClean backwardWeights_native
#axiomsClean backwardDP_native
#axiomsClean backwardDS_native
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
