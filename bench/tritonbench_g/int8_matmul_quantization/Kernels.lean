import VeriTile.Triton.DSL.ImplicitAtomicAdd
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 28000

def quantizeSignature : CallSignature where
  pointers := [("fpa_ptr",.fp32),("a_ptr",.i8),("as_ptr",.fp16)]
  scalars := [("M",.integer),("K",.integer),("stride_fpam",.integer),("stride_fpak",.integer),("stride_am",.integer),("stride_ak",.integer),("stride_asm",.integer)]
  constexpr := ["BLOCK_SIZE_M","BLOCK_SIZE_K"]
  readPorts := ["fpa_ptr"]
  writePorts := ["a_ptr","as_ptr"]

def quantizeKernel
    (fpa_ptr a_ptr as_ptr : RegionName)
    (M K stride_fpam stride_fpak stride_am stride_ak stride_asm BLOCK_SIZE_M BLOCK_SIZE_K : Nat) : ComputeKernel :=
  with_triton_call quantizeSignature call_args [M := M, K := K, stride_fpam := stride_fpam, stride_fpak := stride_fpak, stride_am := stride_am, stride_ak := stride_ak, stride_asm := stride_asm] in triton {
    M = tl.cast(M, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_fpam = tl.cast(stride_fpam, tl.int64)
    stride_fpak = tl.cast(stride_fpak, tl.int64)
    stride_am = tl.cast(stride_am, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_asm = tl.cast(stride_asm, tl.int64)
    pid_m = tl.program_id(axis=0).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M

    fpa_ptrs = fpa_ptr + offs_am[:, None] * stride_fpam + offs_k[None, :] * stride_fpak
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        fpa = tl.load(fpa_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        a_max = tl.maximum(a_max, tl.max(tl.abs(fpa), axis=1))
        fpa_ptrs += BLOCK_SIZE_K * stride_fpak
    a_scale = (a_max / 127.)
    fpa_ptrs = fpa_ptr + offs_am[:, None] * stride_fpam + offs_k[None, :] * stride_fpak
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        fpa = tl.load(fpa_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        inta = (fpa / a_scale[:, None]).to(tl.int8)
        tl.store(a_ptrs, inta, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K)
        fpa_ptrs += BLOCK_SIZE_K * stride_fpak
        a_ptrs += BLOCK_SIZE_K * stride_ak
    as_offs = pid_m * BLOCK_SIZE_M * stride_asm + tl.arange(0, BLOCK_SIZE_M)
    tl.store(as_ptr + as_offs, a_scale)
  }

def unwrappedQuantizeKernel
    (fpa_ptr a_ptr as_ptr : RegionName)
    (M K stride_fpam stride_fpak stride_am stride_ak stride_asm BLOCK_SIZE_M BLOCK_SIZE_K : Nat) : ComputeKernel :=
  with_triton_call quantizeSignature call_args [M := M, K := K, stride_fpam := stride_fpam, stride_fpak := stride_fpak, stride_am := stride_am, stride_ak := stride_ak, stride_asm := stride_asm] in triton {
    M = tl.cast(M, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_fpam = tl.cast(stride_fpam, tl.int64)
    stride_fpak = tl.cast(stride_fpak, tl.int64)
    stride_am = tl.cast(stride_am, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_asm = tl.cast(stride_asm, tl.int64)
    pid_m = tl.program_id(axis=0).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))

    fpa_ptrs = fpa_ptr + offs_am[:, None] * stride_fpam + offs_k[None, :] * stride_fpak
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        fpa = tl.load(fpa_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        a_max = tl.maximum(a_max, tl.max(tl.abs(fpa), axis=1))
        fpa_ptrs += BLOCK_SIZE_K * stride_fpak
    a_scale = (a_max / 127.)
    fpa_ptrs = fpa_ptr + offs_am[:, None] * stride_fpam + offs_k[None, :] * stride_fpak
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        fpa = tl.load(fpa_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        inta = (fpa / a_scale[:, None]).to(tl.int8)
        tl.store(a_ptrs, inta, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K)
        fpa_ptrs += BLOCK_SIZE_K * stride_fpak
        a_ptrs += BLOCK_SIZE_K * stride_ak
    as_offs = pid_m * BLOCK_SIZE_M * stride_asm + tl.arange(0, BLOCK_SIZE_M)
    tl.store(as_ptr + as_offs, a_scale)
  }

def gemmHalfHalfSignature : CallSignature where
  pointers := [("a_ptr",.i8),("as_ptr",.fp16),("b_ptr",.i8),("bs_ptr",.fp32),("c_ptr",.fp16)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),("stride_am",.integer),("stride_ak",.integer),("stride_asm",.integer),("stride_bk",.integer),("stride_bn",.integer),("stride_bsn",.integer),("stride_cm",.integer),("stride_cn",.integer)]
  constexpr := ["BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K","GROUP_SIZE_M","SPLIT_K"]
  readPorts := ["a_ptr","as_ptr","b_ptr","bs_ptr","c_ptr"]
  writePorts := ["c_ptr"]

def gemmHalfHalfKernel
    (a_ptr as_ptr b_ptr bs_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call gemmHalfHalfSignature call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_asm := stride_asm, stride_bk := stride_bk, stride_bn := stride_bn, stride_bsn := stride_bsn, stride_cm := stride_cm, stride_cn := stride_cn] in triton {
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    # Tensor offsets remain exact beyond signed32 element indices.
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_am = tl.cast(stride_am, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_asm = tl.cast(stride_asm, tl.int64)
    stride_bk = tl.cast(stride_bk, tl.int64)
    stride_bn = tl.cast(stride_bn, tl.int64)
    stride_bsn = tl.cast(stride_bsn, tl.int64)
    stride_cm = tl.cast(stride_cm, tl.int64)
    stride_cn = tl.cast(stride_cn, tl.int64)
    pid = tl.program_id(axis=0).to(tl.int64)
    pid_sp_k = tl.program_id(axis=1).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    as_ptrs = as_ptr + offs_am * stride_asm
    bs_ptrs = bs_ptr + offs_bn * stride_bsn
    a_scale = tl.load(as_ptrs, mask=offs_am < M, other=0.0)
    b_scale = tl.load(bs_ptrs, mask=offs_bn < N, other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = (accumulator.to(tl.float32) * a_scale[:, None] * b_scale[None, :]).to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)
  }

def unmaskedGemmHalfHalfKernel
    (a_ptr as_ptr b_ptr bs_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call gemmHalfHalfSignature call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_asm := stride_asm, stride_bk := stride_bk, stride_bn := stride_bn, stride_bsn := stride_bsn, stride_cm := stride_cm, stride_cn := stride_cn] in triton {
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    # Tensor offsets remain exact beyond signed32 element indices.
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_am = tl.cast(stride_am, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_asm = tl.cast(stride_asm, tl.int64)
    stride_bk = tl.cast(stride_bk, tl.int64)
    stride_bn = tl.cast(stride_bn, tl.int64)
    stride_bsn = tl.cast(stride_bsn, tl.int64)
    stride_cm = tl.cast(stride_cm, tl.int64)
    stride_cn = tl.cast(stride_cn, tl.int64)
    pid = tl.program_id(axis=0).to(tl.int64)
    pid_sp_k = tl.program_id(axis=1).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    as_ptrs = as_ptr + offs_am * stride_asm
    bs_ptrs = bs_ptr + offs_bn * stride_bsn
    a_scale = tl.load(as_ptrs)
    b_scale = tl.load(bs_ptrs)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = (accumulator.to(tl.float32) * a_scale[:, None] * b_scale[None, :]).to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)
  }

def gemmHalfSingleSignature : CallSignature where
  pointers := [("a_ptr",.i8),("as_ptr",.fp16),("b_ptr",.i8),("bs_ptr",.fp32),("c_ptr",.fp32)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),("stride_am",.integer),("stride_ak",.integer),("stride_asm",.integer),("stride_bk",.integer),("stride_bn",.integer),("stride_bsn",.integer),("stride_cm",.integer),("stride_cn",.integer)]
  constexpr := ["BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K","GROUP_SIZE_M","SPLIT_K"]
  readPorts := ["a_ptr","as_ptr","b_ptr","bs_ptr","c_ptr"]
  writePorts := ["c_ptr"]

def gemmHalfSingleKernel
    (a_ptr as_ptr b_ptr bs_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call gemmHalfSingleSignature call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_asm := stride_asm, stride_bk := stride_bk, stride_bn := stride_bn, stride_bsn := stride_bsn, stride_cm := stride_cm, stride_cn := stride_cn] in triton {
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    # Tensor offsets remain exact beyond signed32 element indices.
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_am = tl.cast(stride_am, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_asm = tl.cast(stride_asm, tl.int64)
    stride_bk = tl.cast(stride_bk, tl.int64)
    stride_bn = tl.cast(stride_bn, tl.int64)
    stride_bsn = tl.cast(stride_bsn, tl.int64)
    stride_cm = tl.cast(stride_cm, tl.int64)
    stride_cn = tl.cast(stride_cn, tl.int64)
    pid = tl.program_id(axis=0).to(tl.int64)
    pid_sp_k = tl.program_id(axis=1).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    as_ptrs = as_ptr + offs_am * stride_asm
    bs_ptrs = bs_ptr + offs_bn * stride_bsn
    a_scale = tl.load(as_ptrs, mask=offs_am < M, other=0.0)
    b_scale = tl.load(bs_ptrs, mask=offs_bn < N, other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = (accumulator.to(tl.float32) * a_scale[:, None] * b_scale[None, :]).to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)
  }

def unmaskedGemmHalfSingleKernel
    (a_ptr as_ptr b_ptr bs_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call gemmHalfSingleSignature call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_asm := stride_asm, stride_bk := stride_bk, stride_bn := stride_bn, stride_bsn := stride_bsn, stride_cm := stride_cm, stride_cn := stride_cn] in triton {
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    # Tensor offsets remain exact beyond signed32 element indices.
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_am = tl.cast(stride_am, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_asm = tl.cast(stride_asm, tl.int64)
    stride_bk = tl.cast(stride_bk, tl.int64)
    stride_bn = tl.cast(stride_bn, tl.int64)
    stride_bsn = tl.cast(stride_bsn, tl.int64)
    stride_cm = tl.cast(stride_cm, tl.int64)
    stride_cn = tl.cast(stride_cn, tl.int64)
    pid = tl.program_id(axis=0).to(tl.int64)
    pid_sp_k = tl.program_id(axis=1).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    as_ptrs = as_ptr + offs_am * stride_asm
    bs_ptrs = bs_ptr + offs_bn * stride_bsn
    a_scale = tl.load(as_ptrs)
    b_scale = tl.load(bs_ptrs)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = (accumulator.to(tl.float32) * a_scale[:, None] * b_scale[None, :]).to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)
  }

def gemmSingleHalfSignature : CallSignature where
  pointers := [("a_ptr",.i8),("as_ptr",.fp32),("b_ptr",.i8),("bs_ptr",.fp32),("c_ptr",.fp16)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),("stride_am",.integer),("stride_ak",.integer),("stride_asm",.integer),("stride_bk",.integer),("stride_bn",.integer),("stride_bsn",.integer),("stride_cm",.integer),("stride_cn",.integer)]
  constexpr := ["BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K","GROUP_SIZE_M","SPLIT_K"]
  readPorts := ["a_ptr","as_ptr","b_ptr","bs_ptr","c_ptr"]
  writePorts := ["c_ptr"]

def gemmSingleHalfKernel
    (a_ptr as_ptr b_ptr bs_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call gemmSingleHalfSignature call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_asm := stride_asm, stride_bk := stride_bk, stride_bn := stride_bn, stride_bsn := stride_bsn, stride_cm := stride_cm, stride_cn := stride_cn] in triton {
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    # Tensor offsets remain exact beyond signed32 element indices.
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_am = tl.cast(stride_am, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_asm = tl.cast(stride_asm, tl.int64)
    stride_bk = tl.cast(stride_bk, tl.int64)
    stride_bn = tl.cast(stride_bn, tl.int64)
    stride_bsn = tl.cast(stride_bsn, tl.int64)
    stride_cm = tl.cast(stride_cm, tl.int64)
    stride_cn = tl.cast(stride_cn, tl.int64)
    pid = tl.program_id(axis=0).to(tl.int64)
    pid_sp_k = tl.program_id(axis=1).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    as_ptrs = as_ptr + offs_am * stride_asm
    bs_ptrs = bs_ptr + offs_bn * stride_bsn
    a_scale = tl.load(as_ptrs, mask=offs_am < M, other=0.0)
    b_scale = tl.load(bs_ptrs, mask=offs_bn < N, other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = (accumulator.to(tl.float32) * a_scale[:, None] * b_scale[None, :]).to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)
  }

def unmaskedGemmSingleHalfKernel
    (a_ptr as_ptr b_ptr bs_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call gemmSingleHalfSignature call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_asm := stride_asm, stride_bk := stride_bk, stride_bn := stride_bn, stride_bsn := stride_bsn, stride_cm := stride_cm, stride_cn := stride_cn] in triton {
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    # Tensor offsets remain exact beyond signed32 element indices.
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_am = tl.cast(stride_am, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_asm = tl.cast(stride_asm, tl.int64)
    stride_bk = tl.cast(stride_bk, tl.int64)
    stride_bn = tl.cast(stride_bn, tl.int64)
    stride_bsn = tl.cast(stride_bsn, tl.int64)
    stride_cm = tl.cast(stride_cm, tl.int64)
    stride_cn = tl.cast(stride_cn, tl.int64)
    pid = tl.program_id(axis=0).to(tl.int64)
    pid_sp_k = tl.program_id(axis=1).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    as_ptrs = as_ptr + offs_am * stride_asm
    bs_ptrs = bs_ptr + offs_bn * stride_bsn
    a_scale = tl.load(as_ptrs)
    b_scale = tl.load(bs_ptrs)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = (accumulator.to(tl.float32) * a_scale[:, None] * b_scale[None, :]).to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)
  }

def gemmSingleSingleSignature : CallSignature where
  pointers := [("a_ptr",.i8),("as_ptr",.fp32),("b_ptr",.i8),("bs_ptr",.fp32),("c_ptr",.fp32)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),("stride_am",.integer),("stride_ak",.integer),("stride_asm",.integer),("stride_bk",.integer),("stride_bn",.integer),("stride_bsn",.integer),("stride_cm",.integer),("stride_cn",.integer)]
  constexpr := ["BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K","GROUP_SIZE_M","SPLIT_K"]
  readPorts := ["a_ptr","as_ptr","b_ptr","bs_ptr","c_ptr"]
  writePorts := ["c_ptr"]

def gemmSingleSingleKernel
    (a_ptr as_ptr b_ptr bs_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call gemmSingleSingleSignature call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_asm := stride_asm, stride_bk := stride_bk, stride_bn := stride_bn, stride_bsn := stride_bsn, stride_cm := stride_cm, stride_cn := stride_cn] in triton {
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    # Tensor offsets remain exact beyond signed32 element indices.
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_am = tl.cast(stride_am, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_asm = tl.cast(stride_asm, tl.int64)
    stride_bk = tl.cast(stride_bk, tl.int64)
    stride_bn = tl.cast(stride_bn, tl.int64)
    stride_bsn = tl.cast(stride_bsn, tl.int64)
    stride_cm = tl.cast(stride_cm, tl.int64)
    stride_cn = tl.cast(stride_cn, tl.int64)
    pid = tl.program_id(axis=0).to(tl.int64)
    pid_sp_k = tl.program_id(axis=1).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    as_ptrs = as_ptr + offs_am * stride_asm
    bs_ptrs = bs_ptr + offs_bn * stride_bsn
    a_scale = tl.load(as_ptrs, mask=offs_am < M, other=0.0)
    b_scale = tl.load(bs_ptrs, mask=offs_bn < N, other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = (accumulator.to(tl.float32) * a_scale[:, None] * b_scale[None, :]).to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)
  }

def unmaskedGemmSingleSingleKernel
    (a_ptr as_ptr b_ptr bs_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call gemmSingleSingleSignature call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_asm := stride_asm, stride_bk := stride_bk, stride_bn := stride_bn, stride_bsn := stride_bsn, stride_cm := stride_cm, stride_cn := stride_cn] in triton {
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    # Tensor offsets remain exact beyond signed32 element indices.
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_am = tl.cast(stride_am, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_asm = tl.cast(stride_asm, tl.int64)
    stride_bk = tl.cast(stride_bk, tl.int64)
    stride_bn = tl.cast(stride_bn, tl.int64)
    stride_bsn = tl.cast(stride_bsn, tl.int64)
    stride_cm = tl.cast(stride_cm, tl.int64)
    stride_cn = tl.cast(stride_cn, tl.int64)
    pid = tl.program_id(axis=0).to(tl.int64)
    pid_sp_k = tl.program_id(axis=1).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    as_ptrs = as_ptr + offs_am * stride_asm
    bs_ptrs = bs_ptr + offs_bn * stride_bsn
    a_scale = tl.load(as_ptrs)
    b_scale = tl.load(bs_ptrs)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = (accumulator.to(tl.float32) * a_scale[:, None] * b_scale[None, :]).to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)
  }

end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
