import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Mathematical backward source used by the dtype-erasure proofs.
The retired plain-forward proof is replaced by the complete ForwardPublic contracts. -/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton
open VeriTile.Triton

/-- Surface transcription of `ksoftmax_triton.py`'s `_softmax_backward`.

This covers the tested backward paths for `LOG=true/false` and
`CAUSAL=true/false`, preserving the masked gradient/out loads and the two
gradient formulas. -/
def ksoftmax_backward_surface
    (GradIn GradOut Out : RegionName)
    (stride_bm stride_bn stride_gm stride_gn stride_om stride_on K DEPTH : Nat)
    (LOG CAUSAL IS_FP16 : Bool) :
    ComputeKernel := triton {
  m = tl.program_id(0)
  n = tl.program_id(1)
  k = tl.arange(0, $(DEPTH))
  grad_out_ptrs = GradOut + m * $(stride_gm) + n * $(stride_gn) + k
  out_ptrs = Out + m * $(stride_om) + n * $(stride_on) + k
  io_mask = k < $(K)
  if CAUSAL {
    io_mask = io_mask & (k <= n)
  }
  g = tl.load(grad_out_ptrs, mask=io_mask, other=0.0)
  o = tl.load(out_ptrs, mask=io_mask, other=0.0)
  if CAUSAL {
    zero = 0.0
    zero = (zero).to(g.dtype)
    g = tl.where(k > n, zero, g)
    o = tl.where(k > n, zero, o)
  }
  if LOG {
    s = tl.sum(g, 0)
    if IS_FP16 {
      o = (o).to(tl.float32)
    }
    grad_in = g - tl.exp(o) * s
  } else {
    s = tl.sum(g * o, 0)
    grad_in = o * (g - s)
  }
  grad_in_ptrs = GradIn + m * $(stride_bm) + n * $(stride_bn) + k
  tl.store(grad_in_ptrs, grad_in, mask=k < $(K))
}

#axiomsClean ksoftmax_backward_surface
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton
