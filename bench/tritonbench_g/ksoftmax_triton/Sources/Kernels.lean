import VeriTile.Triton.DSL

/-! Complete readable repaired FP32 source. Native.lean separately interprets
widened integer/address operations. Arithmetic, flags and reduction expressions
are unchanged; the causal gradient's final selection is the source repair. -/
namespace VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Triton

/-- Complete original FP32 forward source, including every flag and mask case. -/
def forward (Y X M : RegionName)
    (stride_ym stride_yn stride_xm stride_xn stride_m K DEPTH : Nat)
    (LOG : Bool) (MASK_TYPE : Option String) (CAUSAL IS_FP16 : Bool) : ComputeKernel :=
  let HAS_MASK : Bool := MASK_TYPE.isSome
  let MASK_QK : Bool := MASK_TYPE == some "qk"
  let MASK_BK : Bool := MASK_TYPE == some "bk"
  triton {
  tl.region X = tl.float32, M = tl.float32, Y = tl.float32
  m = tl.program_id(0)
  n = tl.program_id(1)
  k = tl.arange(0, $(DEPTH))
  x_ptrs = X + m * $(stride_xm) + n * $(stride_xn) + k
  io_mask = k < $(K)
  if CAUSAL {
    io_mask = io_mask & (k <= n)
  }
  x = tl.load(x_ptrs, mask=io_mask, other=float("-inf"))
  if CAUSAL {
    off = float("-inf")
    off = (off).to(x.dtype)
    x = tl.where(k > n, off, x)
  }
  if HAS_MASK {
    if MASK_QK {
      mask_ptrs = M + n * $(stride_m) + k
    } elif MASK_BK {
      mask_ptrs = M + m * $(stride_m) + k
    }
    add_mask = tl.load(mask_ptrs, io_mask, other=float("-inf"))
    x += add_mask
  }
  z = x - tl.max(x, axis=0)
  if IS_FP16 {
    z = (z).to(tl.float32)
  }
  num = tl.exp(z)
  denom = tl.sum(num, axis=0)
  if LOG {
    y = z - tl.log(denom)
  } else {
    y = num / denom
  }
  y_ptrs = Y + m * $(stride_ym) + n * $(stride_yn) + k
  tl.store(y_ptrs, (y).to(tl.float32), mask=k < $(K))
}


/-- Complete repaired backward source, with the original LOG/CAUSAL branches. -/
def backward (GradIn GradOut Out : RegionName)
    (stride_bm stride_bn stride_gm stride_gn stride_om stride_on K DEPTH : Nat)
    (LOG CAUSAL : Bool) : ComputeKernel :=
  triton {
  tl.region GradOut = tl.float32, Out = tl.float32, GradIn = tl.float32
  m = tl.program_id(0)
  n = tl.program_id(1)
  k = tl.arange(0, $(DEPTH))
  grad_out_ptrs = GradOut + m * $(stride_gm) + n * $(stride_gn) + k
  out_ptrs = Out + m * $(stride_om) + n * $(stride_on) + k
  io_mask = k < $(K)
  if CAUSAL {
    io_mask = io_mask & (k <= n)
  }
  g = tl.load(grad_out_ptrs, mask=io_mask, other=float(0))
  o = tl.load(out_ptrs, mask=io_mask, other=float(0))
  if CAUSAL {
    zero = float(0)
    zero = (zero).to(g.dtype)
    g = tl.where(k > n, zero, g)
    o = tl.where(k > n, zero, o)
  }
  if LOG {
    s = tl.sum(g, 0)
    grad_in = g - tl.exp(o) * s
  } else {
    s = tl.sum(g * o, 0)
    grad_in = o * (g - s)
  }
  if CAUSAL {
    grad_in = tl.where(k <= n, grad_in, 0.0)
  }
  grad_in_ptrs = GradIn + m * $(stride_bm) + n * $(stride_bn) + k
  tl.store(grad_in_ptrs, (grad_in).to(tl.float32), mask=k < $(K))
 }

end VeriTile.Bench.Repairs.Ksoftmax
