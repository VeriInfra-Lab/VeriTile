import VeriTile.Triton.DSL.CallSignature

/-! New native Python sources derived from the independent affine LayerNorm
specification; no historical Python origin is claimed. FP32 input/parameters
and BF16 output storage are external ABI metadata. Both complete algorithms
retain separate arithmetic, source count casts and all input rereads. -/
namespace VeriTile.Bench.Examples.FusedLayerNorm.Kernels
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("xReg",.fp32),("γReg",.fp32),("βReg",.fp32),("yReg",.bf16)]
  scalars := [("rowStride",.integer)]
  constexpr := ["N","ε"]
  readPorts := ["xReg","γReg","βReg"]
  writePorts := ["yReg"]

def twoPassLayerNormKernel (xReg γReg βReg yReg : RegionName) (N rowStride : Nat) (ε : ℝ) : ComputeKernel :=
  with_triton_call callSignature call_args [rowStride := rowStride] in
  triton {
    pid = tl.program_id(0)
    offs = pid * rowStride + tl.arange(0, N)
    x = tl.load(xReg + offs)
    count = tl.full((), N, tl.uint64).to(tl.float32)
    s_x = tl.sum(x, axis=0)
    μ = s_x / count
    d = x - μ
    s_d2 = tl.sum(d * d, axis=0)
    v = s_d2 / count
    γ = tl.load(γReg + tl.arange(0, N))
    β = tl.load(βReg + tl.arange(0, N))
    σ_inv = 1 / tl.sqrt(v + ε)
    y = (x - μ) * σ_inv * γ + β
    tl.store(yReg + offs, y.to(tl.bfloat16))
  }

def fusedLayerNormKernel (xReg γReg βReg yReg : RegionName) (N rowStride : Nat) (ε : ℝ) : ComputeKernel :=
  with_triton_call callSignature call_args [rowStride := rowStride] in
  triton {
    pid = tl.program_id(0)
    M = 0.0
    S = 0.0
    for i in range(N):
        xi = tl.load(xReg + (pid * rowStride + i))
        delta = xi - M
        M = M + delta / (i.to(tl.float32) + 1)
        delta2 = xi - M
        S = S + delta * delta2
    μ = M
    count = tl.full((), N, tl.uint64).to(tl.float32)
    v = S / count
    σ_inv = 1 / tl.sqrt(v + ε)
    # Read x again to compute every affine output using the shared statistics.
    offs = pid * rowStride + tl.arange(0, N)
    x = tl.load(xReg + offs)
    γ = tl.load(γReg + tl.arange(0, N))
    β = tl.load(βReg + tl.arange(0, N))
    y = (x - μ) * σ_inv * γ + β
    tl.store(yReg + offs, y.to(tl.bfloat16))
  }

end VeriTile.Bench.Examples.FusedLayerNorm.Kernels
