import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelProgramMemory

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicRead
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (ForwardConfig BackwardConfig DotPrimitive queryChunk keyChunk valueChunk)
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

/-- Equality is needed only at the actual checked matrix coordinates. Exterior
memory and untyped cells are unrestricted by this logical input connection. -/
def MatrixEqual (s t : State α) (r : RegionName) (base rows cols rs cs : Nat) : Prop :=
  ∀ row col, row<rows → col<cols →
    (s.mem r (base+row*rs+col*cs)).read .real =
      (t.mem r (base+row*rs+col*cs)).read .real

theorem matrixRead_eq (M : Algebra α) (s t : State α) (r : RegionName)
    (base rows cols rs cs ro co h w : Nat) (same : MatrixEqual s t r base rows cols rs cs) :
    matrixRead M s r base rows cols rs cs ro co h w=
      matrixRead M t r base rows cols rs cs ro co h w := by
  funext i
  simp only [matrixRead]
  split_ifs with active
  · rw [same _ _ active.1 active.2]
  · rfl

theorem forward_values (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig)
    (s t : State α) (pids : s.pids=t.pids)
    (query : MatrixEqual s t c.q (t.pids 2*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d)
    (key : MatrixEqual s t c.k (t.pids 2*c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t)
    (value : MatrixEqual s t c.v (t.pids 2*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d) :
    FPStages.forwardValues M D c s=FPStages.forwardValues M D c t := by
  have qread := matrixRead_eq M s t c.q (t.pids 2*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
  have kread := matrixRead_eq M s t c.k (t.pids 2*c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t
  have vread := matrixRead_eq M s t c.v (t.pids 2*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
  have qeq : forwardQuery M c s=forwardQuery M c t := by
    funext i
    simp only [forwardQuery,queryChunk,keyChunk,pids,fun ro co h w => qread ro co h w query]
  have update : FPStages.forwardUpdate M D c s=FPStages.forwardUpdate M D c t := by
    funext diagonal start previous
    simp only [FPStages.forwardUpdate,queryChunk,keyChunk,valueChunk,pids,qeq,
      fun ro co h w => kread ro co h w key,fun ro co h w => vread ro co h w value]
  have between : FPStages.betweenValues M c s=FPStages.betweenValues M c t := by
    funext previous i
    simp only [FPStages.betweenValues,pids]
  simp only [FPStages.forwardValues,queryChunk,pids,update,between]

theorem query_values (M : Algebra α) (D : DotPrimitive α) (c : BackwardConfig)
    (s t : State α)
    (gradient : MatrixEqual s t c.do_ (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d)
    (key : MatrixEqual s t c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d)
    (value : MatrixEqual s t c.v (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t) :
    FPQuery.values M D c s=FPQuery.values M D c t := by
  have gread := matrixRead_eq M s t c.do_ (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
  have kread := matrixRead_eq M s t c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
  have vread := matrixRead_eq M s t c.v (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t
  have update : FPQuery.update M D c s=FPQuery.update M D c t := by
    funext diagonal start previous
    simp only [FPQuery.update,FPQuery.outputGradient,
      fun ro co h w => gread ro co h w gradient,
      fun ro co h w => kread ro co h w key,fun ro co h w => vread ro co h w value]
  simp only [FPQuery.values,update]

theorem backward_values (M : Algebra α) (D : DotPrimitive α) (c : BackwardConfig)
    (s t : State α)
    (query : MatrixEqual s t c.q (c.i_bh*c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t)
    (gradient : MatrixEqual s t c.do_ (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t)
    (key : MatrixEqual s t c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d)
    (value : MatrixEqual s t c.v (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d) :
    FPStages.backwardValues M D c s=FPStages.backwardValues M D c t := by
  have qread := matrixRead_eq M s t c.q (c.i_bh*c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t
  have gread := matrixRead_eq M s t c.do_ (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t
  have kread := matrixRead_eq M s t c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
  have vread := matrixRead_eq M s t c.v (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
  have update : FPStages.backwardUpdate M D c s=FPStages.backwardUpdate M D c t := by
    funext diagonal start previous
    simp only [FPStages.backwardUpdate,FPStages.decayedKey,backwardKey,backwardValue,
      fun ro co h w => qread ro co h w query,fun ro co h w => gread ro co h w gradient,
      fun ro co h w => kread ro co h w key,fun ro co h w => vread ro co h w value]
  simp only [FPStages.backwardValues,update]

#axiomsClean forward_values
#axiomsClean query_values
#axiomsClean backward_values
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicRead
