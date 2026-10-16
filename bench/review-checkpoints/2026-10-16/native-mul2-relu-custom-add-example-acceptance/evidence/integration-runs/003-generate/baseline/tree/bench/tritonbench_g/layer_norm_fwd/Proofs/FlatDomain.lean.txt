import bench.tritonbench_g.layer_norm_fwd.Proofs.InputDomain
import VeriTile.Triton.Memory.TraceProofs

/-! Finite active-access bounds for the original launch geometry. Input row
strides may overlap. Inactive arguments require no allocation; the second
rowscale/seed/mask stream is bounded independently at M + row. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.FlatDomain
open VeriTile.Triton
noncomputable section

def span (M N stride : Nat) : Nat := (M-1) * stride + N

structure Extents (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (bounds : RegionBounds) : Prop where
  x : span M N stride.x ≤ bounds r.x
  y : span M N stride.y ≤ bounds r.y
  weight : N ≤ bounds r.weight
  rstd : M ≤ bounds r.rstd
  bias : f.bias = Bool.true → N ≤ bounds r.bias
  mean : f.rms = Bool.false → M ≤ bounds r.mean
  residual : f.residual = Bool.true → span M N stride.residual ≤ bounds r.residual
  residualOut : f.residualOut = Bool.true → span M N stride.residualOut ≤ bounds r.residualOut
  x1 : f.x1 = Bool.true → span M N stride.x1 ≤ bounds r.x1
  weight1 : f.w1 = Bool.true → N ≤ bounds r.weight1
  bias1 : f.w1 = Bool.true → f.b1 = Bool.true → N ≤ bounds r.bias1
  y1 : f.w1 = Bool.true → span M N stride.y1 ≤ bounds r.y1
  rowscale : f.rowscale = Bool.true → M ≤ bounds r.rowscale
  rowscale1 : f.x1 = Bool.true → f.rowscale = Bool.true → 2*M ≤ bounds r.rowscale
  seeds : f.dropout = Bool.true → M ≤ bounds r.seeds
  seeds1 : f.x1 = Bool.true → f.dropout = Bool.true → 2*M ≤ bounds r.seeds
  mask : f.dropout = Bool.true → f.dropoutMask = Bool.true → M*N ≤ bounds r.dropoutMask
  mask1 : f.x1 = Bool.true → f.dropout = Bool.true → f.dropoutMask = Bool.true →
    2*M*N ≤ bounds r.dropoutMask

/-- Bounds in the coordinates used by each actual load/store expression. -/
structure RowBounds (f : Flags) (r : Regions) (stride : Strides) (M N row : Nat)
    (bounds : RegionBounds) : Prop where
  x : ∀ col, col < N → row*stride.x+col < bounds r.x
  y : ∀ col, col < N → row*stride.y+col < bounds r.y
  weight : ∀ col, col < N → col < bounds r.weight
  rstd : row < bounds r.rstd
  bias : f.bias = Bool.true → ∀ col, col < N → col < bounds r.bias
  mean : f.rms = Bool.false → row < bounds r.mean
  residual : f.residual = Bool.true → ∀ col, col < N → row*stride.residual+col < bounds r.residual
  residualOut : f.residualOut = Bool.true → ∀ col, col < N → row*stride.residualOut+col < bounds r.residualOut
  x1 : f.x1 = Bool.true → ∀ col, col < N → row*stride.x1+col < bounds r.x1
  weight1 : f.w1 = Bool.true → ∀ col, col < N → col < bounds r.weight1
  bias1 : f.w1 = Bool.true → f.b1 = Bool.true → ∀ col, col < N → col < bounds r.bias1
  y1 : f.w1 = Bool.true → ∀ col, col < N → row*stride.y1+col < bounds r.y1
  rowscale : f.rowscale = Bool.true → row < bounds r.rowscale
  rowscale1 : f.x1 = Bool.true → f.rowscale = Bool.true → M+row < bounds r.rowscale
  seeds : f.dropout = Bool.true → row < bounds r.seeds
  seeds1 : f.x1 = Bool.true → f.dropout = Bool.true → M+row < bounds r.seeds
  mask : f.dropout = Bool.true → f.dropoutMask = Bool.true →
    ∀ col, col < N → row*N+col < bounds r.dropoutMask
  mask1 : f.x1 = Bool.true → f.dropout = Bool.true → f.dropoutMask = Bool.true →
    ∀ col, col < N → (M+row)*N+col < bounds r.dropoutMask

theorem Extents.rowBounds (f : Flags) (r : Regions) (stride : Strides) (M N row : Nat)
    (bounds : RegionBounds) (h : Extents f r stride M N bounds) (hrow : row < M) :
    RowBounds f r stride M N row bounds where
  x := fun col hc => lt_of_lt_of_le (Domain.row_address_lt_span hrow hc) h.x
  y := fun col hc => lt_of_lt_of_le (Domain.row_address_lt_span hrow hc) h.y
  weight := fun col hc => lt_of_lt_of_le hc h.weight
  rstd := lt_of_lt_of_le hrow h.rstd
  bias := fun hf col hc => lt_of_lt_of_le hc (h.bias hf)
  mean := fun hf => lt_of_lt_of_le hrow (h.mean hf)
  residual := fun hf col hc => lt_of_lt_of_le (Domain.row_address_lt_span hrow hc) (h.residual hf)
  residualOut := fun hf col hc => lt_of_lt_of_le (Domain.row_address_lt_span hrow hc) (h.residualOut hf)
  x1 := fun hf col hc => lt_of_lt_of_le (Domain.row_address_lt_span hrow hc) (h.x1 hf)
  weight1 := fun hf col hc => lt_of_lt_of_le hc (h.weight1 hf)
  bias1 := fun hf hb col hc => lt_of_lt_of_le hc (h.bias1 hf hb)
  y1 := fun hf col hc => lt_of_lt_of_le (Domain.row_address_lt_span hrow hc) (h.y1 hf)
  rowscale := fun hf => lt_of_lt_of_le hrow (h.rowscale hf)
  rowscale1 := fun hx hf => lt_of_lt_of_le (Domain.second_row_lt hrow) (h.rowscale1 hx hf)
  seeds := fun hf => lt_of_lt_of_le hrow (h.seeds hf)
  seeds1 := fun hx hf => lt_of_lt_of_le (Domain.second_row_lt hrow) (h.seeds1 hx hf)
  mask := fun hd hm col hc => lt_of_lt_of_le (Domain.first_mask_lt hrow hc) (h.mask hd hm)
  mask1 := fun hx hd hm col hc => lt_of_lt_of_le (Domain.second_mask_lt hrow hc) (h.mask1 hx hd hm)

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.FlatDomain
