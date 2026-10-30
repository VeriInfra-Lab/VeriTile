import VeriTile.Triton.DSL.CallSignature
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Contract

/-! Complete original Python function bodies, including all constexpr branches.
The integer-CSE candidate exposes the quotient already used for group_id in
its remainder expression. The selected live-FMA profile and all FP operations,
loops, pointers, mask shapes and allow_tf32=False are preserved verbatim. -/
namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Source
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

structure Parameters where
  (A B O : RegionName)
  (M N K TM TN TK GROUP : Nat)
  (divM divN divK : Bool)

def callSignature : CallSignature where
  pointers := [("A",.fp32),("B",.fp32),("O",.fp32)]
  scalars := [("M",.i32),("N",.i32),("K",.i32)]
  constexpr := ["TILE_M","TILE_N","TILE_K","GROUP_M","DIVISIBLE_M","DIVISIBLE_N","DIVISIBLE_K"]
  readPorts := ["A","B"]
  writePorts := ["O"]

def kernel (p : Parameters) : ComputeKernel :=
  let A := p.A
  let B := p.B
  let O := p.O
  let TILE_M := p.TM
  let TILE_N := p.TN
  let TILE_K := p.TK
  let GROUP_M := p.GROUP
  let DIVISIBLE_M := p.divM
  let DIVISIBLE_N := p.divN
  let DIVISIBLE_K := p.divK
  with_triton_call callSignature call_args [M := p.M, N := p.N, K := p.K] in triton {
    # batch offsets
    pid_b = tl.program_id(2)
    A += pid_b * M * K
    B += pid_b * K * N
    O += pid_b * M * N

    pidx = tl.program_id(0)
    pidy = tl.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        # reorder CTAs
        gridx = tl.num_programs(0)
        gridy = tl.num_programs(1)
        pid = pidx + pidy * gridx

        num_CTA_per_group = gridy * GROUP_M

        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        if (group_id * GROUP_M + GROUP_M) > gridx:
            GROUP_SIZE = gridx % GROUP_M
        else:
            GROUP_SIZE = GROUP_M
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    o_ptrs = O + offs_m[:, None] * N + offs_n[None, :]

    num_iters = tl.cdiv(K, TILE_K)
    o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for _ in range(num_iters):
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                mask_a = None
            else:
                mask_a = mask_m[:, None]
            if DIVISIBLE_N:
                mask_b = None
            else:
                mask_b = mask_n[None, :]
        else:
            mask_k = offs_k < K
            if DIVISIBLE_M:
                mask_a = mask_k[None, :]
            else:
                mask_a = mask_m[:, None] & mask_k[None, :]
            if DIVISIBLE_N:
                mask_b = mask_k[:, None]
            else:
                mask_b = mask_k[:, None] & mask_n[None, :]

        if DIVISIBLE_M and DIVISIBLE_K:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask_a, other=0.0)
        if DIVISIBLE_N and DIVISIBLE_K:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask_b, other=0.0)

        offs_k += TILE_K
        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

        o += tl.dot(a, b, allow_tf32=False)

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M and not DIVISIBLE_N:
        mask_c = mask_n[None, :]
    elif not DIVISIBLE_M and DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(o_ptrs, o, mask_c)
  }

def quotientRemainder (p : Parameters) : ComputeKernel :=
  let A := p.A
  let B := p.B
  let O := p.O
  let TILE_M := p.TM
  let TILE_N := p.TN
  let TILE_K := p.TK
  let GROUP_M := p.GROUP
  let DIVISIBLE_M := p.divM
  let DIVISIBLE_N := p.divN
  let DIVISIBLE_K := p.divK
  with_triton_call callSignature call_args [M := p.M, N := p.N, K := p.K] in triton {
    # batch offsets
    pid_b = tl.program_id(2)
    A += pid_b * M * K
    B += pid_b * K * N
    O += pid_b * M * N

    pidx = tl.program_id(0)
    pidy = tl.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        # reorder CTAs
        gridx = tl.num_programs(0)
        gridy = tl.num_programs(1)
        pid = pidx + pidy * gridx

        num_CTA_per_group = gridy * GROUP_M

        group_id = pid // num_CTA_per_group
        inner_group_id = pid - (pid // num_CTA_per_group) * num_CTA_per_group
        if (group_id * GROUP_M + GROUP_M) > gridx:
            GROUP_SIZE = gridx % GROUP_M
        else:
            GROUP_SIZE = GROUP_M
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    o_ptrs = O + offs_m[:, None] * N + offs_n[None, :]

    num_iters = tl.cdiv(K, TILE_K)
    o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for _ in range(num_iters):
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                mask_a = None
            else:
                mask_a = mask_m[:, None]
            if DIVISIBLE_N:
                mask_b = None
            else:
                mask_b = mask_n[None, :]
        else:
            mask_k = offs_k < K
            if DIVISIBLE_M:
                mask_a = mask_k[None, :]
            else:
                mask_a = mask_m[:, None] & mask_k[None, :]
            if DIVISIBLE_N:
                mask_b = mask_k[:, None]
            else:
                mask_b = mask_k[:, None] & mask_n[None, :]

        if DIVISIBLE_M and DIVISIBLE_K:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask_a, other=0.0)
        if DIVISIBLE_N and DIVISIBLE_K:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask_b, other=0.0)

        offs_k += TILE_K
        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

        o += tl.dot(a, b, allow_tf32=False)

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M and not DIVISIBLE_N:
        mask_c = mask_n[None, :]
    elif not DIVISIBLE_M and DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(o_ptrs, o, mask_c)
  }

def parameters (s : BmmOptimized.Shape) (c : BmmOptimized.Config) : Parameters :=
  ⟨"A","B","O",s.m,s.n,s.k,c.tm,c.tn,32,c.group,
    decide (s.m%c.tm=0),decide (s.n%c.tn=0),decide (s.k%32=0)⟩

end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Source
