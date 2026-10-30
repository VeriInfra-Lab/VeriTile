"""New spec-derived native HyperConnections source, not recovered historical Python.
Includes full matrix dimensions, every Sinkhorn iteration, scalar specialization
and the existing candidates. All buffers and runtime tau are FP32. Launch with
enable_fp_fusion=False; dot uses input_precision="ieee". The Lean FP rules describe
the declared software model and do not claim GPU/compiler conformance.
"""
import triton
import triton.language as tl

@triton.jit
def matrixKernel(resMixReg, branchOutReg, hPostReg, outReg, tau, S: tl.constexpr, T: tl.constexpr, D: tl.constexpr, numIters: tl.constexpr):
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

@triton.jit
def scalarKernel(resMixReg, branchOutReg, hPostReg, outReg, tau):
    b = tl.program_id(0)
    res_mix = tl.load(resMixReg + b)
    branch_out = tl.load(branchOutReg + b)
    h_post = tl.load(hPostReg)
    branch_mix = tl.exp(h_post / tau) * branch_out
    out = res_mix + branch_mix
    tl.store(outReg + b, out)

@triton.jit
def optimizedKernel(resMixReg, branchOutReg, hPostReg, outReg, tau):
    b = tl.program_id(0)
    res_mix = tl.load(resMixReg + b)
    branch_out = tl.load(branchOutReg + b)
    h_post = tl.load(hPostReg)
    branch_mix = tl.exp(h_post / tau) * branch_out
    out = branch_mix + res_mix
    tl.store(outReg + b, out)

@triton.jit
def matrixOptimized(resMixReg, branchOutReg, hPostReg, outReg, tau, S: tl.constexpr, T: tl.constexpr, D: tl.constexpr, numIters: tl.constexpr):
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

