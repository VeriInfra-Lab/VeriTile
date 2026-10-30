import VeriTile.Triton.DSL

/-! Explicit Triton lowerings. Native source/backend correspondence is an
unsolved goal in Proofs/Backend.lean; compiling these sources does not prove it. -/

namespace VeriTile.Bench.BugDetection.ben_ce_vocab_parallel
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Kernels

def original : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let target_ptr : RegionName := "target"
  let out_ptr : RegionName := "out"
  triton {
    r = tl.arange($(64))
    j = tl.arange($(1024))
    x = tl.load($(x_ptr) + r[:, None] * $(1024) + j[None, :], dtype=tl.bfloat16)
    x = x.to(tl.float32)
    target = tl.load($(target_ptr) + r, dtype=tl.uint32)
    xt = tl.load($(x_ptr) + r * $(1024) + target, dtype=tl.bfloat16)
    xt = xt.to(tl.float32)
    m = tl.max(x, axis=1)
    z = x - m[:, None]
    logs = tl.log(tl.sum(tl.exp(z), axis=1))
    y = m + logs - xt
    tl.store($(out_ptr) + r, y.to(tl.bfloat16))
  }

def candidate : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let target_ptr : RegionName := "target"
  let out_ptr : RegionName := "out"
  triton {
    r = tl.arange($(64))
    j = tl.arange($(1024))
    x = tl.load($(x_ptr) + r[:, None] * $(1024) + j[None, :], dtype=tl.bfloat16)
    x = x.to(tl.float32)
    target = tl.load($(target_ptr) + r, dtype=tl.uint32)
    xt = tl.load($(x_ptr) + r * $(1024) + target, dtype=tl.bfloat16)
    xt = xt.to(tl.float32)
    k = tl.arange($(512))
    a = tl.load($(x_ptr) + r[:, None] * $(1024) + k[None, :], dtype=tl.bfloat16)
    a = a.to(tl.float32)
    b = tl.load($(x_ptr) + r[:, None] * $(1024) + $(512) + k[None, :], dtype=tl.bfloat16)
    b = b.to(tl.float32)
    ma = tl.max(a, axis=1)
    mb = tl.max(b, axis=1)
    la = ma + tl.log(tl.sum(tl.exp(a - ma[:, None]), axis=1))
    lb = mb + tl.log(tl.sum(tl.exp(b - mb[:, None]), axis=1))
    m = tl.maximum(la, lb)
    lse = m + tl.log(tl.exp(la - m) + tl.exp(lb - m))
    y = lse - xt
    tl.store($(out_ptr) + r, y.to(tl.bfloat16))
  }

end Kernels
end
end VeriTile.Bench.BugDetection.ben_ce_vocab_parallel
