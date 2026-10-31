import VeriTile.Triton.DSL.CallSignature

/-! New native sources derived from the independent softmax formula.
No historical Python origin is claimed. Call metadata carries FP32 storage;
source scalar initializers keep the complete online loop in FP32. Both input
passes and every output store are in the authoritative raw Python body.
libdevice.exp retains the admitted intrinsic; the source graph is separate
arithmetic with enable_fp_fusion=False. -/
namespace VeriTile.Bench.Examples.OnlineSoftmax.Kernels
open VeriTile Triton

def batchSignature : CallSignature where
  pointers := [("xReg",.fp32),("yReg",.fp32)]
  scalars := []
  constexpr := ["blockSize"]
  readPorts := ["xReg"]
  writePorts := ["yReg"]

def onlineSignature : CallSignature where
  pointers := [("xReg",.fp32),("yReg",.fp32)]
  scalars := []
  constexpr := ["N"]
  readPorts := ["xReg"]
  writePorts := ["yReg"]

def stableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature batchSignature in
  triton {
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    m = tl.max(x, axis=0)
    e = libdevice.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(yReg + offs, y)
  }

def onlineSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_signature onlineSignature in
  triton {
    pid = tl.program_id(0)
    m = tl.full((), float("-inf"), tl.float32)
    l = tl.full((), 0, tl.float32)
    for i in range(N):
        xi = tl.load(xReg + (pid * N + i))
        m_new = tl.maximum(m, xi)
        l = libdevice.exp(m - m_new) * l + libdevice.exp(xi - m_new)
        m = m_new
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    x = tl.load(xReg + offs)
    e = libdevice.exp(x - m)
    y = e / l
    tl.store(yReg + offs, y)
  }

/-- The batch entry shares the exact stable source above. -/
def batchSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : ComputeKernel :=
  stableSoftmaxKernel xReg yReg N

end VeriTile.Bench.Examples.OnlineSoftmax.Kernels
