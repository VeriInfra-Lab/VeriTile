"""New spec-derived native HyperConnections source, not recovered historical Python.
Includes full matrix dimensions, every Sinkhorn iteration, scalar specialization
and the existing candidates. All buffers and runtime tau are FP32. Launch with
enable_fp_fusion=False; dot uses input_precision="ieee". The Lean FP rules describe
the declared software model and do not claim GPU/compiler conformance.
"""
import triton
import triton.language as tl

@triton.jit
def matrixKernel(resReg, hResReg, hPreReg, resMixReg, branchInReg, tau, S: tl.constexpr, T: tl.constexpr, D: tl.constexpr, numIters: tl.constexpr):
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

@triton.jit
def scalarKernel(resReg, hResReg, hPreReg, resMixReg, branchInReg, tau):
    b = tl.program_id(0)
    residual = tl.load(resReg + b)
    h_res = tl.load(hResReg)
    res_mix = tl.exp(h_res / tau) * residual
    h_pre = tl.load(hPreReg)
    branch_in = tl.exp(h_pre / tau) * residual
    tl.store(resMixReg + b, res_mix)
    tl.store(branchInReg + b, branch_in)

@triton.jit
def middleKernel(resReg, hResReg, hPreReg, resMixReg, branchInReg, tau):
    b = tl.program_id(0)
    residual = tl.load(resReg + b)
    h_res = tl.load(hResReg)
    res_mix = residual * tl.exp(h_res / tau)
    h_pre = tl.load(hPreReg)
    branch_in = tl.exp(h_pre / tau) * residual
    tl.store(resMixReg + b, res_mix)
    tl.store(branchInReg + b, branch_in)

@triton.jit
def optimizedKernel(resReg, hResReg, hPreReg, resMixReg, branchInReg, tau):
    b = tl.program_id(0)
    residual = tl.load(resReg + b)
    h_res = tl.load(hResReg)
    res_mix = residual * tl.exp(h_res / tau)
    h_pre = tl.load(hPreReg)
    branch_in = residual * tl.exp(h_pre / tau)
    tl.store(resMixReg + b, res_mix)
    tl.store(branchInReg + b, branch_in)

@triton.jit
def matrixMiddle(resReg, hResReg, hPreReg, resMixReg, branchInReg, tau, S: tl.constexpr, T: tl.constexpr, D: tl.constexpr, numIters: tl.constexpr):
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

@triton.jit
def matrixOptimized(resReg, hResReg, hPreReg, resMixReg, branchInReg, tau, S: tl.constexpr, T: tl.constexpr, D: tl.constexpr, numIters: tl.constexpr):
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

