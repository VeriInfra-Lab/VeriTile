import VeriTile.Triton.DSL.ClosedUnaryJitFunctions
namespace VeriTile.Bench.Repairs.TritonLinearActivation
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedVariables false

def tanhSource : ClosedUnaryJitFunction := with_triton_params (x) constexpr_args [] in triton {
    """Tanh activation function"""
    return tl.extra.cuda.libdevice.tanh(x)
}

def reluSource : ClosedUnaryJitFunction := with_triton_params (x) constexpr_args [] in triton {
    """Relu activation function"""
    return tl.maximum(0, x)
}

def fast_geluSource : ClosedUnaryJitFunction := with_triton_params (x) constexpr_args [] in triton {
    """Fast approximation of the gelu function. May slightly decrease accuracy."""
    return 0.5 * x * (1 + tanh(sqrt2pi * (x + 0.044715 * x * x * x)))
}

def geluSource : ClosedUnaryJitFunction := with_triton_params (x) constexpr_args [] in triton {
    """Gaussian Error Linear Unit (GELU)"""
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x / sqrt2))
}

def signature (dtype : FloatDType) : CallSignature where
  pointers := []
  floatingPointers := [("C",dtype), ("ACT_INPUTS",dtype), ("A",dtype), ("B",dtype), ("bias",dtype)]
  scalars := [("M",.integer), ("N",.integer), ("K",.integer), ("CACHE_KEY_M",.integer), ("CACHE_KEY_N",.integer), ("CACHE_KEY_K",.integer), ("output_m_stride",.integer), ("output_n_stride",.integer), ("act_inputs_m_stride",.integer), ("act_inputs_n_stride",.integer), ("a_m_stride",.integer), ("a_k_stride",.integer), ("b_n_stride",.integer), ("b_k_stride",.integer)]
  constexpr := ["BLOCK_M", "GROUP_M", "BLOCK_N", "BLOCK_K", "SPLIT_K", "K_LOAD_MASK_NEEDED", "HAS_BIAS", "SHOULD_SAVE_ACT_INPUTS", "ACTIVATION"]
  sourceConstants := [("sqrt2pi","math.sqrt(2.0 / math.pi)"),("sqrt2","math.sqrt(2.0)")]
  readPorts := ["A","B","bias"]
  writePorts := ["C","ACT_INPUTS"]

def originalKernel (dtype : FloatDType) (C ACT_INPUTS A B bias : RegionName) (M N K CACHE_KEY_M CACHE_KEY_N CACHE_KEY_K output_m_stride output_n_stride act_inputs_m_stride act_inputs_n_stride a_m_stride a_k_stride b_n_stride b_k_stride BLOCK_M GROUP_M BLOCK_N BLOCK_K SPLIT_K : Nat) (HAS_BIAS SHOULD_SAVE_ACT_INPUTS K_LOAD_MASK_NEEDED : Bool) (ACTIVATION : Option String) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [M := M, N := N, K := K, CACHE_KEY_M := CACHE_KEY_M, CACHE_KEY_N := CACHE_KEY_N, CACHE_KEY_K := CACHE_KEY_K, output_m_stride := output_m_stride, output_n_stride := output_n_stride, act_inputs_m_stride := act_inputs_m_stride, act_inputs_n_stride := act_inputs_n_stride, a_m_stride := a_m_stride, a_k_stride := a_k_stride, b_n_stride := b_n_stride, b_k_stride := b_k_stride]
    jit_functions [tanh := tanhSource, relu := reluSource, fast_gelu := fast_geluSource, gelu := geluSource] in triton {
    """
    Kernel for computing Out = activation(A @ W.T + bias)

    - Input has shape (M, K)
    - Weight has shape (N, K)
    - Bias has shape (N,)
    - Output has shape (M, N)
    - ActInputs (optional) has shape (M, N)

    'ActInputs' optionally saves the A x W + C intermediate for backward computations

    This kernel will consolidate over K
    """
    program_idx = tl.program_id(axis=0)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_idx = program_idx // width
    group_size = min(grid_m - group_idx * GROUP_M, GROUP_M)
    block_m_idx = group_idx * GROUP_M + (program_idx % group_size)
    block_n_idx = (program_idx % width) // group_size

    # now compute the block that each program will go through
    # m_offs (resp. n_offs) denotes a range of indices
    # for rows (resp. col) of C
    m_offs_untagged = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs_untagged = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    # trick to avoid masking on M and N axis
    # m_offs_untagged and n_offs_untagged can contains addresses outside matrix boundaries
    # modulo operation is used to wrap around the indices that go beyond the matrix boundaries
    # The value loaded are not ok but at least we are not reading outside the A/B matrices
    # Then, during storing in C a mask is used and the results related to these wrong values is discarded!
    # Wrapped tails are not contiguous, so no false alignment/contiguity hints are attached.
    m_offs = m_offs_untagged % M
    n_offs = n_offs_untagged % N

    k_range_offs = tl.arange(0, BLOCK_K)

    A = A + (m_offs[:, None] * a_m_stride + k_range_offs[None, :] * a_k_stride)
    B = B + (k_range_offs[:, None] * b_k_stride + n_offs[None, :] * b_n_stride)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if HAS_BIAS:
        bias = tl.load(bias + n_offs, mask=n_offs < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    for k in range(K, 0, -BLOCK_K):
        if K_LOAD_MASK_NEEDED:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=k_range_offs[None, :] < k, other=0.0)
            b = tl.load(B, mask=k_range_offs[:, None] < k, other=0.0)
        acc += tl.dot(a, b)

        A += BLOCK_K * a_k_stride
        B += BLOCK_K * b_k_stride

    # optional: save the activation inputs
    if SHOULD_SAVE_ACT_INPUTS:
        act_in_ptrs = ACT_INPUTS + m_offs_untagged[:, None] * act_inputs_m_stride + n_offs_untagged[None, :] * act_inputs_n_stride
        save_mask = (m_offs_untagged < M)[:, None] & (n_offs_untagged < N)[None, :]
        tl.store(act_in_ptrs, acc, mask=save_mask)

    # optional: fused activation (while the data is in shared memory)
    if ACTIVATION == "tanh":
        acc = tanh(acc)
    if ACTIVATION == "gelu":
        acc = gelu(acc)
    if ACTIVATION == "fast_gelu":
        acc = fast_gelu(acc)
    if ACTIVATION == "relu":
        acc = relu(acc)

    # write back result
    C = C + m_offs_untagged[:, None] * output_m_stride + n_offs_untagged[None, :] * output_n_stride
    c_ptr_mask = (m_offs_untagged < M)[:, None] & (n_offs_untagged < N)[None, :]
    tl.store(C, acc, mask=c_ptr_mask)
}

def candidateKernel (dtype : FloatDType) (C ACT_INPUTS A B bias : RegionName) (M N K CACHE_KEY_M CACHE_KEY_N CACHE_KEY_K output_m_stride output_n_stride act_inputs_m_stride act_inputs_n_stride a_m_stride a_k_stride b_n_stride b_k_stride BLOCK_M GROUP_M BLOCK_N BLOCK_K SPLIT_K : Nat) (HAS_BIAS SHOULD_SAVE_ACT_INPUTS K_LOAD_MASK_NEEDED : Bool) (ACTIVATION : Option String) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [M := M, N := N, K := K, CACHE_KEY_M := CACHE_KEY_M, CACHE_KEY_N := CACHE_KEY_N, CACHE_KEY_K := CACHE_KEY_K, output_m_stride := output_m_stride, output_n_stride := output_n_stride, act_inputs_m_stride := act_inputs_m_stride, act_inputs_n_stride := act_inputs_n_stride, a_m_stride := a_m_stride, a_k_stride := a_k_stride, b_n_stride := b_n_stride, b_k_stride := b_k_stride]
    jit_functions [tanh := tanhSource, relu := reluSource, fast_gelu := fast_geluSource, gelu := geluSource] in triton {
    """
    Kernel for computing Out = activation(A @ W.T + bias)

    - Input has shape (M, K)
    - Weight has shape (N, K)
    - Bias has shape (N,)
    - Output has shape (M, N)
    - ActInputs (optional) has shape (M, N)

    'ActInputs' optionally saves the A x W + C intermediate for backward computations

    This kernel will consolidate over K
    """
    program_idx = tl.program_id(axis=0)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    if GROUP_M == 8:
        width = grid_n << 3
    else:
        width = GROUP_M * grid_n
    group_idx = program_idx // width
    group_size = min(grid_m - group_idx * GROUP_M, GROUP_M)
    block_m_idx = group_idx * GROUP_M + (program_idx % group_size)
    block_n_idx = (program_idx % width) // group_size

    # now compute the block that each program will go through
    # m_offs (resp. n_offs) denotes a range of indices
    # for rows (resp. col) of C
    m_offs_untagged = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs_untagged = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    # trick to avoid masking on M and N axis
    # m_offs_untagged and n_offs_untagged can contains addresses outside matrix boundaries
    # modulo operation is used to wrap around the indices that go beyond the matrix boundaries
    # The value loaded are not ok but at least we are not reading outside the A/B matrices
    # Then, during storing in C a mask is used and the results related to these wrong values is discarded!
    # Wrapped tails are not contiguous, so no false alignment/contiguity hints are attached.
    m_offs = m_offs_untagged % M
    n_offs = n_offs_untagged % N

    k_range_offs = tl.arange(0, BLOCK_K)

    A = A + (m_offs[:, None] * a_m_stride + k_range_offs[None, :] * a_k_stride)
    B = B + (k_range_offs[:, None] * b_k_stride + n_offs[None, :] * b_n_stride)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if HAS_BIAS:
        bias = tl.load(bias + n_offs, mask=n_offs < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    for k in range(K, 0, -BLOCK_K):
        if K_LOAD_MASK_NEEDED:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=k_range_offs[None, :] < k, other=0.0)
            b = tl.load(B, mask=k_range_offs[:, None] < k, other=0.0)
        acc += tl.dot(a, b)

        A += BLOCK_K * a_k_stride
        B += BLOCK_K * b_k_stride

    # optional: save the activation inputs
    if SHOULD_SAVE_ACT_INPUTS:
        act_in_ptrs = ACT_INPUTS + m_offs_untagged[:, None] * act_inputs_m_stride + n_offs_untagged[None, :] * act_inputs_n_stride
        save_mask = (m_offs_untagged < M)[:, None] & (n_offs_untagged < N)[None, :]
        tl.store(act_in_ptrs, acc, mask=save_mask)

    # optional: fused activation (while the data is in shared memory)
    if ACTIVATION == "tanh":
        acc = tanh(acc)
    if ACTIVATION == "gelu":
        acc = gelu(acc)
    if ACTIVATION == "fast_gelu":
        acc = fast_gelu(acc)
    if ACTIVATION == "relu":
        acc = relu(acc)

    # write back result
    C = C + m_offs_untagged[:, None] * output_m_stride + n_offs_untagged[None, :] * output_n_stride
    c_ptr_mask = (m_offs_untagged < M)[:, None] & (n_offs_untagged < N)[None, :]
    tl.store(C, acc, mask=c_ptr_mask)
}

end
end VeriTile.Bench.Repairs.TritonLinearActivation
