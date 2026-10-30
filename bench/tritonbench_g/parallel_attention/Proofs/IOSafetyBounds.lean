import bench.tritonbench_g.parallel_attention.QueryWhole
import bench.tritonbench_g.parallel_attention.FPBackwardWhole
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOSafety
open VeriTile.Triton FP.Structural
noncomputable section

/-- Only active two-dimensional windows need backing. Offsets of masked
block-pointer lanes may lie outside these rectangles. -/
def MatrixBounds (bounds : RegionName → Nat) (region : RegionName)
    (base rows cols rowStride colStride : Nat) : Prop :=
  ∀ row col,row<rows → col<cols → base+row*rowStride+col*colStride<bounds region

/-- Output bounds cover the actual tile lanes. Native descriptors may have
K greater than BK; no uncomputed feature is silently added to the output. -/
def TileBounds (bounds : RegionName → Nat) (region : RegionName)
    (base rows cols rowStride colStride rowOffset colOffset m n : Nat) : Prop :=
  ∀ (i : Fin m) (j : Fin n),rowOffset+i.val<rows → colOffset+j.val<cols →
    base+(rowOffset+i.val)*rowStride+(colOffset+j.val)*colStride<bounds region

structure ForwardBounds {α : Type} (c : ForwardConfig) (s : State α)
    (bounds : RegionName → Nat) : Prop where
  query : MatrixBounds bounds c.q (s.pids 2*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
  key : MatrixBounds bounds c.k (s.pids 2*c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t
  value : MatrixBounds bounds c.v (s.pids 2*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
  output : TileBounds bounds c.o
    ((s.pids 2+c.B*c.H*(s.pids 0/((c.V+c.BV-1)/c.BV)))*c.s_vo_h)
    c.T c.V c.s_vo_t c.s_vo_d (queryChunk c s) (valueChunk c s) c.BTL c.BV
  normalizer : s.pids 0%((c.V+c.BV-1)/c.BV)=0 → ∀ (i : Fin c.BTL),queryChunk c s+i.val<c.T → (s.pids 2+c.B*c.H*(s.pids 0/((c.V+c.BV-1)/c.BV)))*c.T+(queryChunk c s+i.val)<bounds c.z

structure BackwardBounds (c : BackwardConfig) (bounds : RegionName → Nat) : Prop where
  query : MatrixBounds bounds c.q (c.i_bh*c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t
  key : MatrixBounds bounds c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
  value : MatrixBounds bounds c.v (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
  outGrad : MatrixBounds bounds c.do_ (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t
  normGrad : ∀ row,row<c.T → c.i_bh*c.T+row<bounds c.dz
  keyGrad : TileBounds bounds c.dk ((c.i_bh+c.B*c.H*c.i_v)*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
    (c.i_c*c.BTL) (c.i_k*c.BK) c.BTL c.BK
  valueGrad : TileBounds bounds c.dv ((c.i_bh+c.B*c.H*c.i_k)*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
    (c.i_c*c.BTL) (c.i_v*c.BV) c.BTL c.BV

structure QueryBounds (c : QueryConfig) (bounds : RegionName → Nat) : Prop where
  query : MatrixBounds bounds c.q (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
  key : MatrixBounds bounds c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
  value : MatrixBounds bounds c.v (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t
  outGrad : MatrixBounds bounds c.do_ (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
  normGrad : ∀ row,row<c.T → c.i_bh*c.T+row<bounds c.dz
  queryGrad : TileBounds bounds c.dq ((c.i_bh+c.B*c.H*c.i_v)*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
    (c.i_c*c.BTL) (c.i_k*c.BK) c.BTL c.BK

end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOSafety
