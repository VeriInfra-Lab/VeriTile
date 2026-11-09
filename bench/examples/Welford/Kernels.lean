import VeriTile.Triton.DSL.CallSignature

/-! Native sources authored from the independent population mean/variance
specifications. No historical Python origin is claimed; see native_kernels.py.
FP32 input and BF16 scalar output storage are explicit in the external ABI.
Both output addresses are zero: these are per-program row contracts.
Separate multiply/add semantics use enable_fp_fusion=False. -/
namespace VeriTile.Bench.Examples.Welford.Kernels
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("xReg",.fp32),("meanReg",.bf16),("varReg",.bf16)]
  scalars := [("rowStride",.integer)]
  constexpr := ["blockSize"]

def twopassWelfordKernel (xReg meanReg varReg : RegionName) (blockSize rowStride : Nat) : ComputeKernel :=
  with_triton_call callSignature call_args [rowStride := rowStride] in
  triton {
    pid = tl.program_id(0)
    offs = pid * rowStride + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    count = tl.full((), blockSize, tl.uint64).to(tl.float32)
    s_x = tl.sum(x, axis=0)
    μ = s_x / count
    d = x - μ
    s_d2 = tl.sum(d * d, axis=0)
    v = s_d2 / count
    tl.store(meanReg, μ.to(tl.bfloat16))
    tl.store(varReg, v.to(tl.bfloat16))
  }

def onlineWelfordKernel (xReg meanReg varReg : RegionName) (blockSize rowStride : Nat) : ComputeKernel :=
  with_triton_call callSignature call_args [rowStride := rowStride] in
  triton {
    pid = tl.program_id(0)
    M = 0.0
    S = 0.0
    for i in range(blockSize):
        xi = tl.load(xReg + (pid * rowStride + i))
        delta = xi - M
        M = M + delta / (i.to(tl.float32) + 1)
        delta2 = xi - M
        S = S + delta * delta2
    count = tl.full((), blockSize, tl.uint64).to(tl.float32)
    tl.store(meanReg, M.to(tl.bfloat16))
    tl.store(varReg, (S / count).to(tl.bfloat16))
  }

end VeriTile.Bench.Examples.Welford.Kernels
