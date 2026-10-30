import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! New spec-derived native Python, not historical source; see source-references.json.
The intrinsic choices, conditional fallbacks and BF16 stores are explicit. -/

namespace VeriTile.Bench.Examples.StableLogSumExp.Kernels
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("xReg",.fp32),("yReg",.bf16)]
  scalars := []
  constexpr := ["blockSize"]

def directLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    e = libdevice.exp(x)
    s = tl.sum(e, axis=0)
    y = tl.log(s)
    tl.store(yReg + pid, y.to(tl.bfloat16))
  }

def stableLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    m = tl.max(x, axis=0)
    e = libdevice.exp(x - m)
    s = tl.sum(e, axis=0)
    y = m + tl.log(s)
    tl.store(yReg + pid, y.to(tl.bfloat16))
  }

def optimizedLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    m = tl.max(x, axis=0)
    e = libdevice.exp(x - m)
    s = tl.sum(e, axis=0)
    p = s * libdevice.exp(m)
    keep = (p >= 0.5) & (p <= 2.0)
    direct = tl.log(tl.where(keep, p, 1.0))
    split_m = tl.where(keep, 0.0, m)
    simplify = (0.5 < tl.abs(split_m)) & (tl.abs(split_m) <= 80.0)
    fallback = tl.log(libdevice.exp(tl.where(simplify, 0.0, split_m)))
    log_em = tl.where(simplify, split_m, fallback)
    split = tl.log(tl.where(keep, 1.0, s)) + log_em
    y = tl.where(keep, direct, split)
    tl.store(yReg + pid, y.to(tl.bfloat16))
  }

end VeriTile.Bench.Examples.StableLogSumExp.Kernels
