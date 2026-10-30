import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOLogical

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOLogical
open VeriTile.Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Finite input locality; padding lanes impose no input equality. -/
def RectangleEq (a b : InputValues α) (R : RegionName)
    (base rows cols rowStride colStride : Nat) : Prop :=
  ∀ row col,row<rows → col<cols →
    a R (base+row*rowStride+col*colStride)=b R (base+row*rowStride+col*colStride)

theorem matrix_congr (M : Algebra α) {a b : InputValues α}
    {R : RegionName} {base rows cols rowStride colStride : Nat}
    (h : RectangleEq a b R base rows cols rowStride colStride) (ro co m n : Nat) :
    matrixRead M a R base rows cols rowStride colStride ro co m n =
      matrixRead M b R base rows cols rowStride colStride ro co m n := by
  funext i
  dsimp only [matrixRead]
  split
  next active => rw [h _ _ active.1 active.2]
  next => rfl

theorem raw_matrix_congr (M : Algebra α) {a b : InputValues α}
    {R : RegionName} {base rows cols rowStride colStride : Nat}
    (h : RectangleEq a b R base rows cols rowStride colStride) (ro co m n : Nat) :
    rawMatrixRead M a R base rows cols rowStride colStride ro co m n =
      rawMatrixRead M b R base rows cols rowStride colStride ro co m n := by
  funext i
  dsimp only [rawMatrixRead]
  split
  next active => exact h _ _ active.1 active.2
  next => rfl

structure ForwardEq (c : ForwardConfig) (pid : Nat → Nat) (a b : InputValues α) : Prop where
  query : RectangleEq a b c.q (pid 2*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
  key : RectangleEq a b c.k (pid 2*c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t
  value : RectangleEq a b c.v (pid 2*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d

structure BackwardEq (c : BackwardConfig) (a b : InputValues α) : Prop where
  query : RectangleEq a b c.q (c.i_bh*c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t
  key : RectangleEq a b c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
  value : RectangleEq a b c.v (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
  outGrad : RectangleEq a b c.do_ (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t
  normGrad : ∀ row,row<c.T → a c.dz (c.i_bh*c.T+row)=b c.dz (c.i_bh*c.T+row)

structure QueryEq (c : QueryConfig) (a b : InputValues α) : Prop where
  query : RectangleEq a b c.q (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
  key : RectangleEq a b c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
  value : RectangleEq a b c.v (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t
  outGrad : RectangleEq a b c.do_ (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
  normGrad : ∀ row,row<c.T → a c.dz (c.i_bh*c.T+row)=b c.dz (c.i_bh*c.T+row)

theorem forward_congr (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig)
    (pid : Nat → Nat) (a b : InputValues α) (h : ForwardEq c pid a b) :
    forwardValues M D c pid a=forwardValues M D c pid b := by
  have hq : forwardQuery M c pid a=forwardQuery M c pid b := by
    funext i
    simp only [forwardQuery,matrix_congr M h.query]
  have hu (diag : Bool) : forwardUpdate M D c pid a diag=forwardUpdate M D c pid b diag := by
    funext start acc
    simp only [forwardUpdate,hq,
      matrix_congr M h.key,matrix_congr M h.value]
  simp only [forwardValues,hu]

theorem backward_congr (M : Algebra α) (D : DotPrimitive α) (c : BackwardConfig)
    (a b : InputValues α) (h : BackwardEq c a b) :
    backwardValues M D c a=backwardValues M D c b := by
  have hk : backwardKey M c a=backwardKey M c b := by
    funext i
    simp only [backwardKey,matrix_congr M h.key]
  have hv : backwardValue M c a=backwardValue M c b := by
    funext i
    simp only [backwardValue,matrix_congr M h.value]
  have hn (start : Nat) (i : Fin c.BTS) :
      (if start+i.val<c.T then a c.dz (c.i_bh*c.T+start+i.val) else M.literal (some .fp32) .real 0.0)=
      (if start+i.val<c.T then b c.dz (c.i_bh*c.T+start+i.val) else M.literal (some .fp32) .real 0.0) := by
    split
    next active => simpa only [Nat.add_assoc] using h.normGrad _ active
    next => rfl
  have hu (diag : Bool) : backwardUpdate M D c a diag=backwardUpdate M D c b diag := by
    funext start acc
    simp only [backwardUpdate,hk,hv,matrix_congr M h.query,raw_matrix_congr M h.outGrad,hn]
  simp only [backwardValues,hu]

theorem query_congr (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (a b : InputValues α) (h : QueryEq c a b) :
    queryValues M D c a=queryValues M D c b := by
  have hq : queryInput M c a=queryInput M c b := by
    funext i
    simp only [queryInput,matrix_congr M h.query]
  have ho : queryOutGrad M c a=queryOutGrad M c b := by
    funext i
    simp only [queryOutGrad,raw_matrix_congr M h.outGrad]
  have hn : queryNormGrad M c a=queryNormGrad M c b := by
    funext i
    unfold queryNormGrad
    split
    next active => rw [show c.i_bh*c.T+c.i_c*c.BTL+i.1.val=c.i_bh*c.T+(c.i_c*c.BTL+i.1.val) by omega,h.normGrad _ active]
    next => rfl
  have hu (diag : Bool) : queryUpdate M D c a diag=queryUpdate M D c b diag := by
    funext start acc
    simp only [queryUpdate,hq,ho,matrix_congr M h.key,matrix_congr M h.value,hn]
  simp only [queryValues,hu]

#axiomsClean forward_congr
#axiomsClean backward_congr
#axiomsClean query_congr
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOLogical
