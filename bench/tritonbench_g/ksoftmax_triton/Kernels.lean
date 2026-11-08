import VeriTile.Triton.DSL.OptionalStringConstants

/-! Complete existing repaired forward and backward function bodies. Runtime
scalar ABI and constexpr values are supplied outside the quotations. -/
namespace VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Triton

def forwardSignature : CallSignature where
  pointers := [("Y",.fp32), ("X",.fp32), ("M",.fp32)]
  scalars := [("stride_ym",.i64), ("stride_yn",.i64), ("stride_xm",.i64), ("stride_xn",.i64), ("stride_m",.i64), ("K",.i64)]
  constexpr := ["LOG", "MASK_TYPE", "CAUSAL", "DEPTH", "IS_FP16"]
  readPorts := ["X", "M"]
  writePorts := ["Y"]

def forwardKernel (Y X M : RegionName)
    (stride_ym stride_yn stride_xm stride_xn stride_m K DEPTH : Nat)
    (LOG : Bool) (MASK_TYPE : Option String) (CAUSAL IS_FP16 : Bool) : ComputeKernel :=
  with_triton_call forwardSignature call_args [stride_ym := stride_ym, stride_yn := stride_yn, stride_xm := stride_xm, stride_xn := stride_xn, stride_m := stride_m, K := K] in triton {
    """
    Fused softmax kernel over a 3d tensor.
    The softmax is applied over the last dimension, equivalent to torch.softmax(tensor, dim=-1)
    """
    m = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1).to(tl.int64)
    k = tl.arange(0, DEPTH).to(tl.int64)
    x_ptrs = X + m * stride_xm + n * stride_xn + k
    io_mask = k < K
    if CAUSAL:
        io_mask = io_mask & (k <= n)
    x = tl.load(x_ptrs, mask=io_mask, other=float("-inf"))
    if CAUSAL:
        off = float("-inf")
        off = off.to(x.dtype)
        x = tl.where(k > n, off, x)
    if MASK_TYPE is not None:
        if MASK_TYPE == 'qk':
            mask_ptrs = M + n * stride_m + k
        elif MASK_TYPE == 'bk':
            mask_ptrs = M + m * stride_m + k
        add_mask = tl.load(mask_ptrs, io_mask, other=float("-inf"))
        x += add_mask
    z = x - tl.max(x, axis=0)
    if IS_FP16:
        z = z.to(tl.float32)
    num = tl.exp(z)
    denom = tl.sum(num, axis=0)
    if LOG:
        y = z - tl.log(denom)
    else:
        y = num / denom
    y_ptrs = Y + m * stride_ym + n * stride_yn + k
    tl.store(y_ptrs, y, mask=k < K)
  }

def backwardSignature : CallSignature where
  pointers := [("GradIn",.fp32), ("GradOut",.fp32), ("Out",.fp32)]
  scalars := [("stride_bm",.i64), ("stride_bn",.i64), ("stride_gm",.i64), ("stride_gn",.i64), ("stride_om",.i64), ("stride_on",.i64), ("K",.i64)]
  constexpr := ["LOG", "CAUSAL", "DEPTH", "IS_FP16"]
  readPorts := ["GradOut", "Out"]
  writePorts := ["GradIn"]

def backwardTypedKernel (GradIn GradOut Out : RegionName)
    (stride_bm stride_bn stride_gm stride_gn stride_om stride_on K DEPTH : Nat)
    (LOG CAUSAL IS_FP16 : Bool) : ComputeKernel :=
  with_triton_call backwardSignature call_args [stride_bm := stride_bm, stride_bn := stride_bn, stride_gm := stride_gm, stride_gn := stride_gn, stride_om := stride_om, stride_on := stride_on, K := K] in triton {
    """
    Compute the softmax gradients.
    """
    m = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1).to(tl.int64)
    k = tl.arange(0, DEPTH).to(tl.int64)
    grad_out_ptrs = GradOut + m * stride_gm + n * stride_gn + k
    out_ptrs = Out + m * stride_om + n * stride_on + k
    io_mask = k < K
    if CAUSAL:
        io_mask = io_mask & (k <= n)
    g = tl.load(grad_out_ptrs, mask=io_mask, other=float(0))
    o = tl.load(out_ptrs, mask=io_mask, other=float(0))
    if CAUSAL:
        zero = float(0)
        zero = zero.to(g.dtype)
        g = tl.where(k > n, zero, g)
        o = tl.where(k > n, zero, o)
    if LOG:
        s = tl.sum(g, 0)
        if IS_FP16:
            o = o.to(tl.float32)
        grad_in = g - tl.exp(o) * s
    else:
        s = tl.sum(g * o, 0)
        grad_in = o * (g - s)
    if CAUSAL:
        grad_in = tl.where(k <= n, grad_in, 0.)
    grad_in_ptrs = GradIn + m * stride_bm + n * stride_bn + k
    tl.store(grad_in_ptrs, grad_in, mask=k < K)
  }

def backward (GradIn GradOut Out : RegionName)
    (stride_bm stride_bn stride_gm stride_gn stride_om stride_on K DEPTH : Nat)
    (LOG CAUSAL : Bool) : ComputeKernel :=
  backwardTypedKernel GradIn GradOut Out stride_bm stride_bn stride_gm stride_gn stride_om stride_on K DEPTH LOG CAUSAL Bool.false

end VeriTile.Bench.Repairs.Ksoftmax


namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked
/-- The guard-only candidate uses the frozen complete source quotations. -/
abbrev forward := Ksoftmax.forwardKernel
abbrev backward := Ksoftmax.backward

theorem forward_unchanged : forward=Ksoftmax.forwardKernel := rfl
theorem backward_unchanged : backward=Ksoftmax.backward := rfl
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked
