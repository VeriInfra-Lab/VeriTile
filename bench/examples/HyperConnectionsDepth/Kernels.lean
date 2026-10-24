import VeriTile.Triton.DSL.CallSignature

/-! New native Python from the independent full Sinkhorn/matrix specification;
no recovered historical source is asserted. Every dimension and normalization
iteration is retained, alongside the scalar specialization and existing
candidates. Parameters and buffers are FP32; tau is a runtime argument bound
at this invocation, not a Python constexpr. Dot input_precision="ieee" and
separate arithmetic (enable_fp_fusion=False) specify the source graph. -/
namespace VeriTile.Bench.Examples.HyperConnectionsDepth.Kernels
open VeriTile Triton

def matrixKernelSignature : CallSignature where
  pointers := [("resMixReg",.fp32), ("branchOutReg",.fp32), ("hPostReg",.fp32), ("outReg",.fp32)]
  scalars := [("tau",.fp32)]
  constexpr := ["S","T","D","numIters"]

def matrixKernel (resMixReg branchOutReg hPostReg outReg : RegionName) (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  with_triton_call matrixKernelSignature call_args [tau := tau] in
  triton {
    b      = tl.program_id(0)
    offs_s = tl.arange(0, S)
    offs_t = tl.arange(0, T)
    offs_t2 = tl.arange(0, T)
    offs_d = tl.arange(0, D)

    res_mix_ptrs = b * (S * D) + offs_s[:, None] * D + offs_d[None, :]
    branch_ptrs = b * (T * D) + offs_t[:, None] * D + offs_d[None, :]
    res_mix = tl.load(resMixReg + res_mix_ptrs)
    branch_out = tl.load(branchOutReg + branch_ptrs)

    h_post_ptrs = offs_t[:, None] * S + offs_s[None, :]
    h_post_logits = tl.load(hPostReg + h_post_ptrs)
    z = h_post_logits / tau
    u = tl.zeros([T])
    v = tl.zeros([S])
    for iter in range(numIters):
        u = 0 - tl.log(tl.sum(tl.exp(z + v[None, :]), axis = 1))
        v = 0 - tl.log(tl.sum(tl.exp(z + u[:, None]), axis = 0))

    post_weights = tl.exp(z + u[:, None] + v[None, :])
    branch_mix = tl.dot(tl.trans(post_weights), branch_out, input_precision="ieee")
    out = res_mix + branch_mix

    out_ptrs = b * (S * D) + offs_s[:, None] * D + offs_d[None, :]
    tl.store(outReg + out_ptrs, out)
  }

def scalarKernelSignature : CallSignature where
  pointers := [("resMixReg",.fp32), ("branchOutReg",.fp32), ("hPostReg",.fp32), ("outReg",.fp32)]
  scalars := [("tau",.fp32)]
  constexpr := []

def scalarKernel (resMixReg branchOutReg hPostReg outReg : RegionName) (tau : ℝ) : ComputeKernel :=
  with_triton_call scalarKernelSignature call_args [tau := tau] in
  triton {
    b = tl.program_id(0)
    res_mix = tl.load(resMixReg + b)
    branch_out = tl.load(branchOutReg + b)
    h_post = tl.load(hPostReg)
    branch_mix = tl.exp(h_post / tau) * branch_out
    out = res_mix + branch_mix
    tl.store(outReg + b, out)
  }

def optimizedKernelSignature : CallSignature where
  pointers := [("resMixReg",.fp32), ("branchOutReg",.fp32), ("hPostReg",.fp32), ("outReg",.fp32)]
  scalars := [("tau",.fp32)]
  constexpr := []

def optimizedKernel (tau : ℝ) : ComputeKernel :=
  let resMixReg : RegionName := "res_mix"
  let branchOutReg : RegionName := "branch_out"
  let hPostReg : RegionName := "h_post"
  let outReg : RegionName := "out"
  with_triton_call optimizedKernelSignature call_args [tau := tau] in
  triton {
    b = tl.program_id(0)
    res_mix = tl.load(resMixReg + b)
    branch_out = tl.load(branchOutReg + b)
    h_post = tl.load(hPostReg)
    branch_mix = tl.exp(h_post / tau) * branch_out
    out = branch_mix + res_mix
    tl.store(outReg + b, out)
  }

def matrixOptimizedSignature : CallSignature where
  pointers := [("resMixReg",.fp32), ("branchOutReg",.fp32), ("hPostReg",.fp32), ("outReg",.fp32)]
  scalars := [("tau",.fp32)]
  constexpr := ["S","T","D","numIters"]

def matrixOptimized (resMixReg branchOutReg hPostReg outReg : RegionName) (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  with_triton_call matrixOptimizedSignature call_args [tau := tau] in
  triton {
    b      = tl.program_id(0)
    offs_s = tl.arange(0, S)
    offs_t = tl.arange(0, T)
    offs_t2 = tl.arange(0, T)
    offs_d = tl.arange(0, D)

    res_mix_ptrs = b * (S * D) + offs_s[:, None] * D + offs_d[None, :]
    branch_ptrs = b * (T * D) + offs_t[:, None] * D + offs_d[None, :]
    res_mix = tl.load(resMixReg + res_mix_ptrs)
    branch_out = tl.load(branchOutReg + branch_ptrs)

    h_post_ptrs = offs_t[:, None] * S + offs_s[None, :]
    h_post_logits = tl.load(hPostReg + h_post_ptrs)
    z = h_post_logits / tau
    u = tl.zeros([T])
    v = tl.zeros([S])
    for iter in range(numIters):
        u = 0 - tl.log(tl.sum(tl.exp(z + v[None, :]), axis = 1))
        v = 0 - tl.log(tl.sum(tl.exp(z + u[:, None]), axis = 0))

    post_weights = tl.exp(z + u[:, None] + v[None, :])
    branch_mix = tl.dot(tl.trans(post_weights), branch_out, input_precision="ieee")
    out = branch_mix + res_mix

    out_ptrs = b * (S * D) + offs_s[:, None] * D + offs_d[None, :]
    tl.store(outReg + out_ptrs, out)
  }

def mhcDepthConnectionKernel (resMixReg branchOutReg hPostReg outReg : RegionName) (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  match S,T,D,numIters with
  | 1,1,1,0 => scalarKernel resMixReg branchOutReg hPostReg outReg tau
  | _,_,_,_ => matrixKernel resMixReg branchOutReg hPostReg outReg S T D numIters tau

def originalKernel (tau : ℝ) : ComputeKernel := scalarKernel "res_mix" "branch_out" "h_post" "out" tau

def matrixOriginal (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  matrixKernel "res_mix" "branch_out" "h_post" "out" S T D numIters tau
end VeriTile.Bench.Examples.HyperConnectionsDepth.Kernels
