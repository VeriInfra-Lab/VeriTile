import VeriTile.Triton.DSL.SignedByteReads

/-! Exact repaired Python bodies. All invocation typing is outside the body. -/
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Body
open VeriTile.Triton
noncomputable section

def unscaledSignature : CallSignature where
  pointers := [("a_ptr",.i8),("b_ptr",.i8),("c_ptr",.i32)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),("stride_am",.integer),("stride_ak",.integer),("stride_bk",.integer),("stride_bn",.integer),("stride_cm",.integer),("stride_cn",.integer)]
  constexpr := ["BLOCK_M","BLOCK_N","BLOCK_K","GROUP_M"]
  readPorts := ["a_ptr","b_ptr"]
  writePorts := ["c_ptr"]

def unscaledKernel (a_ptr b_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) : ComputeKernel :=
  with_triton_call unscaledSignature call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn] in triton {
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See the matrix multiplication tutorial for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_n = (pid % num_pid_in_group) // GROUP_M

    # ----------------------------------------------------------
    # Create block pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate.
    # See above `Make a Block Pointer` section for details.
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_M, BLOCK_N]` block.
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        # Load with boundary checks, no need to calculate the mask manually.
        # For better performance, you may remove some axis from the boundary
        # check, if you can guarantee that the access is always in-bound in
        # that axis.
        # See above `Load/Store a Block Pointer` section for details.
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # We accumulate along the K dimension.
        # Keep the wrapping add separate from Triton's saturating INT8 MMA.
        accumulator = tl.inline_asm_elementwise(
            "add.u32 $0, $1, $2;", constraints="=r,r,r",
            args=[accumulator, tl.dot(a, b)], dtype=tl.int32, is_pure=True, pack=1,
        )
        # Advance the block pointer to the next K block.
        # See above `Advance a Block Pointer` section for details.
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    c = accumulator  # .to(tl.float16)

    # ----------------------------------------------------------------
    # Write back the block of the output matrix C with boundary checks.
    # See above `Load/Store a Block Pointer` section for details.
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))
  }

def unscaledCandidateKernel (a_ptr b_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) : ComputeKernel :=
  with_triton_call unscaledSignature call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn] in triton {
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See the matrix multiplication tutorial for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_n = (pid % num_pid_in_group) // GROUP_M

    # ----------------------------------------------------------
    # Create block pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate.
    # See above `Make a Block Pointer` section for details.
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_M, BLOCK_N]` block.
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        # Load with boundary checks, no need to calculate the mask manually.
        # For better performance, you may remove some axis from the boundary
        # check, if you can guarantee that the access is always in-bound in
        # that axis.
        # See above `Load/Store a Block Pointer` section for details.
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # We accumulate along the K dimension.
        # Keep the wrapping add separate from Triton's saturating INT8 MMA.
        accumulator = tl.inline_asm_elementwise(
            "add.u32 $0, $1, $2;", constraints="=r,r,r",
            args=[accumulator, tl.dot(a, b)], dtype=tl.int32, is_pure=True, pack=1,
        )
        # Advance the block pointer to the next K block.
        # See above `Advance a Block Pointer` section for details.
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # ----------------------------------------------------------------
    # Write back the block of the output matrix C with boundary checks.
    # See above `Load/Store a Block Pointer` section for details.
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))
  }

def scaledSignature : CallSignature where
  pointers := [("a_ptr",.i8),("b_ptr",.i8),("c_ptr",.i32),("s1_ptr",.fp32)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),("stride_am",.integer),("stride_ak",.integer),("stride_bk",.integer),("stride_bn",.integer),("stride_cm",.integer),("stride_cn",.integer),("stride_s1m",.integer),("stride_s1n",.integer)]
  constexpr := ["BLOCK_M","BLOCK_N","BLOCK_K","GROUP_M","EVEN_K","ACC_TYPE"]
  readPorts := ["a_ptr","b_ptr","s1_ptr"]
  writePorts := ["c_ptr"]

def scaledKernel (a_ptr b_ptr c_ptr s1_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn stride_s1m stride_s1n BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) (EVEN_K : Bool) : ComputeKernel :=
  let ACC_TYPE : SourceDType := .i32
  with_triton_call scaledSignature call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn, stride_s1m := stride_s1m, stride_s1n := stride_s1n] in triton {
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = rm % M
    rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = a_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = b_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        # A zero-seeded tile dot is bounded; the carry update wraps modulo 2**32.
        acc = tl.inline_asm_elementwise(
            "add.u32 $0, $1, $2;", constraints="=r,r,r",
            args=[acc, tl.dot(a, b)], dtype=tl.int32, is_pure=True, pack=1,
        )
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n * stride_cn + idx_m * stride_cm
    tmp0 = tl.load(
        s1_ptr + (tl.broadcast_to(idx_m * stride_s1m, mask.shape)),
        mask,
        eviction_policy="evict_last",
    )
    tl.store(c_ptr + (tl.broadcast_to(xindex, mask.shape)), acc * tmp0, mask)
  }

def scaledCandidateKernel (a_ptr b_ptr c_ptr s1_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn stride_s1m stride_s1n BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) (EVEN_K : Bool) : ComputeKernel :=
  let ACC_TYPE : SourceDType := .i32
  with_triton_call scaledSignature call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn, stride_s1m := stride_s1m, stride_s1n := stride_s1n] in triton {
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = rm % M
    rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = a_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = b_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        # A zero-seeded tile dot is bounded; the carry update wraps modulo 2**32.
        acc = tl.inline_asm_elementwise(
            "add.u32 $0, $1, $2;", constraints="=r,r,r",
            args=[acc, tl.dot(a, b)], dtype=tl.int32, is_pure=True, pack=1,
        )
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # Reuse the unchanged initial rm/rn registers.
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n * stride_cn + idx_m * stride_cm
    tmp0 = tl.load(
        s1_ptr + (tl.broadcast_to(idx_m * stride_s1m, mask.shape)),
        mask,
        eviction_policy="evict_last",
    )
    tl.store(c_ptr + (tl.broadcast_to(xindex, mask.shape)), acc * tmp0, mask)
  }

end
end VeriTile.Bench.Repairs.IntScaledMatmul.Body
