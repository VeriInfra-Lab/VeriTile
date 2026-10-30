import VeriTile.Triton.DSL.CallSignature

/-! New native Python from the independent full Sinkhorn/matrix specification;
no recovered historical source is asserted. Every dimension and normalization
iteration is retained, alongside the scalar specialization and existing
candidates. Parameters and buffers are FP32; tau is a runtime argument bound
at this invocation, not a Python constexpr. Dot input_precision="ieee" and
separate arithmetic (enable_fp_fusion=False) specify the source graph. -/
namespace VeriTile.Bench.Examples.HyperConnectionsWidth.Kernels
open VeriTile Triton

def matrixKernelSignature : CallSignature where
  pointers := [("resReg",.fp32), ("hResReg",.fp32), ("hPreReg",.fp32), ("resMixReg",.fp32), ("branchInReg",.fp32)]
  scalars := [("tau",.fp32)]
  constexpr := ["S","T","D","numIters"]

def matrixKernel (resReg hResReg hPreReg resMixReg branchInReg : RegionName) (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  with_triton_call matrixKernelSignature call_args [tau := tau] in
  triton {
    b      = tl.program_id(0)
    offs_s = tl.arange(0, S)
    offs_s2 = tl.arange(0, S)
    offs_t = tl.arange(0, T)
    offs_d = tl.arange(0, D)

    res_ptrs = b * (S * D) + offs_s[:, None] * D + offs_d[None, :]
    residuals = tl.load(resReg + res_ptrs)

    h_res_ptrs = offs_s[:, None] * S + offs_s2[None, :]
    h_res_logits = tl.load(hResReg + h_res_ptrs)
    z = h_res_logits / tau
    u = tl.zeros([S])
    v = tl.zeros([S])
    for iter in range(numIters):
        u = 0 - tl.log(tl.sum(tl.exp(z + v[None, :]), axis = 1))
        v = 0 - tl.log(tl.sum(tl.exp(z + u[:, None]), axis = 0))

    res_weights = tl.exp(z + u[:, None] + v[None, :])
    res_mix = tl.dot(res_weights, residuals, input_precision="ieee")

    h_pre_ptrs = offs_s[:, None] * T + offs_t[None, :]
    h_pre_logits = tl.load(hPreReg + h_pre_ptrs)
    z = h_pre_logits / tau
    u = tl.zeros([S])
    v = tl.zeros([T])
    for iter in range(numIters):
        u = 0 - tl.log(tl.sum(tl.exp(z + v[None, :]), axis = 1))
        v = 0 - tl.log(tl.sum(tl.exp(z + u[:, None]), axis = 0))

    pre_weights = tl.exp(z + u[:, None] + v[None, :])
    branch_in = tl.dot(tl.trans(pre_weights), residuals, input_precision="ieee")

    res_mix_ptrs = b * (S * D) + offs_s[:, None] * D + offs_d[None, :]
    branch_ptrs = b * (T * D) + offs_t[:, None] * D + offs_d[None, :]
    tl.store(resMixReg + res_mix_ptrs, res_mix)
    tl.store(branchInReg + branch_ptrs, branch_in)
  }

def scalarKernelSignature : CallSignature where
  pointers := [("resReg",.fp32), ("hResReg",.fp32), ("hPreReg",.fp32), ("resMixReg",.fp32), ("branchInReg",.fp32)]
  scalars := [("tau",.fp32)]
  constexpr := []

def scalarKernel (resReg hResReg hPreReg resMixReg branchInReg : RegionName) (tau : ℝ) : ComputeKernel :=
  with_triton_call scalarKernelSignature call_args [tau := tau] in
  triton {
    b = tl.program_id(0)
    residual = tl.load(resReg + b)
    h_res = tl.load(hResReg)
    res_mix = tl.exp(h_res / tau) * residual
    h_pre = tl.load(hPreReg)
    branch_in = tl.exp(h_pre / tau) * residual
    tl.store(resMixReg + b, res_mix)
    tl.store(branchInReg + b, branch_in)
  }

def middleKernelSignature : CallSignature where
  pointers := [("resReg",.fp32), ("hResReg",.fp32), ("hPreReg",.fp32), ("resMixReg",.fp32), ("branchInReg",.fp32)]
  scalars := [("tau",.fp32)]
  constexpr := []

def middleKernel (tau : ℝ) : ComputeKernel :=
  let resReg : RegionName := "res"
  let hResReg : RegionName := "h_res"
  let hPreReg : RegionName := "h_pre"
  let resMixReg : RegionName := "res_mix"
  let branchInReg : RegionName := "branch_in"
  with_triton_call middleKernelSignature call_args [tau := tau] in
  triton {
    b = tl.program_id(0)
    residual = tl.load(resReg + b)
    h_res = tl.load(hResReg)
    res_mix = residual * tl.exp(h_res / tau)
    h_pre = tl.load(hPreReg)
    branch_in = tl.exp(h_pre / tau) * residual
    tl.store(resMixReg + b, res_mix)
    tl.store(branchInReg + b, branch_in)
  }

def optimizedKernelSignature : CallSignature where
  pointers := [("resReg",.fp32), ("hResReg",.fp32), ("hPreReg",.fp32), ("resMixReg",.fp32), ("branchInReg",.fp32)]
  scalars := [("tau",.fp32)]
  constexpr := []

def optimizedKernel (tau : ℝ) : ComputeKernel :=
  let resReg : RegionName := "res"
  let hResReg : RegionName := "h_res"
  let hPreReg : RegionName := "h_pre"
  let resMixReg : RegionName := "res_mix"
  let branchInReg : RegionName := "branch_in"
  with_triton_call optimizedKernelSignature call_args [tau := tau] in
  triton {
    b = tl.program_id(0)
    residual = tl.load(resReg + b)
    h_res = tl.load(hResReg)
    res_mix = residual * tl.exp(h_res / tau)
    h_pre = tl.load(hPreReg)
    branch_in = residual * tl.exp(h_pre / tau)
    tl.store(resMixReg + b, res_mix)
    tl.store(branchInReg + b, branch_in)
  }

def matrixMiddleSignature : CallSignature where
  pointers := [("resReg",.fp32), ("hResReg",.fp32), ("hPreReg",.fp32), ("resMixReg",.fp32), ("branchInReg",.fp32)]
  scalars := [("tau",.fp32)]
  constexpr := ["S","T","D","numIters"]

def matrixMiddle (resReg hResReg hPreReg resMixReg branchInReg : RegionName) (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  with_triton_call matrixMiddleSignature call_args [tau := tau] in
  triton {
    b      = tl.program_id(0)
    offs_s = tl.arange(0, S)
    offs_s2 = tl.arange(0, S)
    offs_t = tl.arange(0, T)
    offs_d = tl.arange(0, D)

    res_ptrs = b * (S * D) + offs_s[:, None] * D + offs_d[None, :]
    residuals = tl.load(resReg + res_ptrs)

    h_res_ptrs = offs_s[:, None] * S + offs_s2[None, :]
    h_res_logits = tl.load(hResReg + h_res_ptrs)
    z = h_res_logits * (1 / tau)
    u = tl.zeros([S])
    v = tl.zeros([S])
    for iter in range(numIters):
        u = 0 - tl.log(tl.sum(tl.exp(z + v[None, :]), axis = 1))
        v = 0 - tl.log(tl.sum(tl.exp(z + u[:, None]), axis = 0))

    res_weights = tl.exp(z + u[:, None] + v[None, :])
    res_mix = tl.dot(res_weights, residuals, input_precision="ieee")

    h_pre_ptrs = offs_s[:, None] * T + offs_t[None, :]
    h_pre_logits = tl.load(hPreReg + h_pre_ptrs)
    z = h_pre_logits / tau
    u = tl.zeros([S])
    v = tl.zeros([T])
    for iter in range(numIters):
        u = 0 - tl.log(tl.sum(tl.exp(z + v[None, :]), axis = 1))
        v = 0 - tl.log(tl.sum(tl.exp(z + u[:, None]), axis = 0))

    pre_weights = tl.exp(z + u[:, None] + v[None, :])
    branch_in = tl.dot(tl.trans(pre_weights), residuals, input_precision="ieee")

    res_mix_ptrs = b * (S * D) + offs_s[:, None] * D + offs_d[None, :]
    branch_ptrs = b * (T * D) + offs_t[:, None] * D + offs_d[None, :]
    tl.store(resMixReg + res_mix_ptrs, res_mix)
    tl.store(branchInReg + branch_ptrs, branch_in)
  }

def matrixOptimizedSignature : CallSignature where
  pointers := [("resReg",.fp32), ("hResReg",.fp32), ("hPreReg",.fp32), ("resMixReg",.fp32), ("branchInReg",.fp32)]
  scalars := [("tau",.fp32)]
  constexpr := ["S","T","D","numIters"]

def matrixOptimized (resReg hResReg hPreReg resMixReg branchInReg : RegionName) (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  with_triton_call matrixOptimizedSignature call_args [tau := tau] in
  triton {
    b      = tl.program_id(0)
    offs_s = tl.arange(0, S)
    offs_s2 = tl.arange(0, S)
    offs_t = tl.arange(0, T)
    offs_d = tl.arange(0, D)

    res_ptrs = b * (S * D) + offs_s[:, None] * D + offs_d[None, :]
    residuals = tl.load(resReg + res_ptrs)

    h_res_ptrs = offs_s[:, None] * S + offs_s2[None, :]
    h_res_logits = tl.load(hResReg + h_res_ptrs)
    z = h_res_logits * (1 / tau)
    u = tl.zeros([S])
    v = tl.zeros([S])
    for iter in range(numIters):
        u = 0 - tl.log(tl.sum(tl.exp(z + v[None, :]), axis = 1))
        v = 0 - tl.log(tl.sum(tl.exp(z + u[:, None]), axis = 0))

    res_weights = tl.exp(z + u[:, None] + v[None, :])
    res_mix = tl.dot(res_weights, residuals, input_precision="ieee")

    h_pre_ptrs = offs_s[:, None] * T + offs_t[None, :]
    h_pre_logits = tl.load(hPreReg + h_pre_ptrs)
    z = h_pre_logits * (1 / tau)
    u = tl.zeros([S])
    v = tl.zeros([T])
    for iter in range(numIters):
        u = 0 - tl.log(tl.sum(tl.exp(z + v[None, :]), axis = 1))
        v = 0 - tl.log(tl.sum(tl.exp(z + u[:, None]), axis = 0))

    pre_weights = tl.exp(z + u[:, None] + v[None, :])
    branch_in = tl.dot(tl.trans(pre_weights), residuals, input_precision="ieee")

    res_mix_ptrs = b * (S * D) + offs_s[:, None] * D + offs_d[None, :]
    branch_ptrs = b * (T * D) + offs_t[:, None] * D + offs_d[None, :]
    tl.store(resMixReg + res_mix_ptrs, res_mix)
    tl.store(branchInReg + branch_ptrs, branch_in)
  }

def mhcWidthConnectionKernel (resReg hResReg hPreReg resMixReg branchInReg : RegionName) (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  match S,T,D,numIters with
  | 1,1,1,0 => scalarKernel resReg hResReg hPreReg resMixReg branchInReg tau
  | _,_,_,_ => matrixKernel resReg hResReg hPreReg resMixReg branchInReg S T D numIters tau

def originalKernel (tau : ℝ) : ComputeKernel := scalarKernel "res" "h_res" "h_pre" "res_mix" "branch_in" tau

def matrixOriginal (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  matrixKernel "res" "h_res" "h_pre" "res_mix" "branch_in" S T D numIters tau
end VeriTile.Bench.Examples.HyperConnectionsWidth.Kernels
