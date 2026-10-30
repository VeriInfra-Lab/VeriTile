from typing import Optional
import math

import torch
import triton
import triton.language as tl
from torch.autograd.function import FunctionCtx
from torch.cuda.amp import custom_fwd
try:
    from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
    _prune_configs_by = {"early_config_prune": early_config_prune, "perf_model": estimate_matmul_time, "top_k": 10}
except ModuleNotFoundError as error:
    if error.name not in ("triton.ops", "triton.ops.matmul_perf_model"):
        raise
    # Current Triton omits this optional legacy performance model.
    # The complete original configuration family remains available to autotuning.
    _prune_configs_by = None


sqrt2pi = tl.constexpr(math.sqrt(2.0 / math.pi))
sqrt2 = tl.constexpr(math.sqrt(2.0))


@triton.jit
def tanh(x):
    """Tanh activation function"""
    return tl.extra.cuda.libdevice.tanh(x)


@triton.jit
def relu(x):
    """Relu activation function"""
    return tl.maximum(0, x)


@triton.jit
def fast_gelu(x):
    """Fast approximation of the gelu function. May slightly decrease accuracy."""
    return 0.5 * x * (1 + tanh(sqrt2pi * (x + 0.044715 * x * x * x)))


@triton.jit
def gelu(x):
    """Gaussian Error Linear Unit (GELU)"""
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x / sqrt2))


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, "SPLIT_K": 1},
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
                    # split_k not used
                    # for split_k in [2, 4, 8, 16]:
                    #     configs.append(triton.Config(
                    #         {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k},
                    #         num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=5, num_warps=2),
        # good for int8
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=5, num_warps=2),
    ]
    + get_configs_io_bound(),
    key=["CACHE_KEY_M", "CACHE_KEY_N", "CACHE_KEY_K"],
    prune_configs_by=_prune_configs_by,
)
@triton.heuristics(
    {
        "K_LOAD_MASK_NEEDED": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def kernel_fma_shifted(
    C,  # Pointers to matrices
    ACT_INPUTS,
    A,
    B,
    bias,
    # Matrix dimensions
    M,
    N,
    K,
    CACHE_KEY_M,
    CACHE_KEY_N,
    CACHE_KEY_K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    output_m_stride,
    output_n_stride,
    act_inputs_m_stride,
    act_inputs_n_stride,
    a_m_stride,
    a_k_stride,
    b_n_stride,
    b_k_stride,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # split k not used, not performant with activation, kept because early_config_prune is expecting it
    SPLIT_K: tl.constexpr,
    K_LOAD_MASK_NEEDED: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SHOULD_SAVE_ACT_INPUTS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
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


def _canonical_fp32(value, device, name):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or not value.is_cuda or value.device != device
            or value.dtype != torch.float32 or not value.is_contiguous()
            or value.storage_offset() != 0 or value.is_neg() or value.is_conj()):
        raise ValueError(name + " must be an ordinary canonical FP32 CUDA tensor")
    if value.numel() and value.data_ptr() % 4 != 0:
        raise ValueError(name + " must have an aligned active FP32 pointer")
    stride = 1
    for axis in range(value.ndim - 1, -1, -1):
        if value.stride(axis) != stride:
            raise ValueError(name + " must have canonical strides")
        stride *= max(1, value.shape[axis])


def _validate_inputs(x, weight, bias, activation, act_inputs):
    if not isinstance(x, torch.Tensor) or x.ndim not in (2, 3):
        raise ValueError("x must be a rank-two matrix or rank-three batch")
    if activation is not None and not isinstance(activation, str):
        raise TypeError("activation must be a string or None")
    _canonical_fp32(x, x.device, "x")
    _canonical_fp32(weight, x.device, "weight")
    if weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError("weight must have shape [N,K] matching x")
    M = x.shape[0] if x.ndim == 2 else x.shape[0] * x.shape[1]
    N, K = weight.shape
    inputs = [x, weight]
    if bias is not None:
        _canonical_fp32(bias, x.device, "bias")
        if bias.shape != (N,):
            raise ValueError("bias must have shape [N]")
        inputs.append(bias)
    if act_inputs is not None:
        _canonical_fp32(act_inputs, x.device, "act_inputs")
        if act_inputs.shape != (M, N):
            raise ValueError("act_inputs must have shape [M,N]")
        inputs.append(act_inputs)
    live = [v for v in inputs if v.numel()]
    for i, left in enumerate(live):
        a = left.untyped_storage()
        for right in live[i + 1:]:
            b = right.untyped_storage()
            if not (a.data_ptr() + a.nbytes() <= b.data_ptr()
                    or b.data_ptr() + b.nbytes() <= a.data_ptr()):
                raise ValueError("input and saved allocations must be independent")
    # Source expressions, including masked output lanes and the final K-pointer
    # increment, remain signed32. Legal runtime launch limits are a separate ABI.
    if (max(M, N, K) + 256 >= 2**31
            or (M + 256) * max(N, 1) + N + 256 >= 2**31
            or M * max(K, 1) + K + 256 >= 2**31
            or N * max(K, 1) + K + 256 >= 2**31):
        raise ValueError("formed source indices exceed the admitted signed32 range")


class _SavedBuffer:
    # AMP recursively casts tensor arguments, but not this private holder.
    # The public forward still receives the original tensor in its usual slot.
    __slots__ = ("tensor",)

    def __init__(self, tensor):
        self.tensor = tensor


class LinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, activation, act_inputs):
        _validate_inputs(x, weight, bias, activation, act_inputs)
        return LinearLayer._forward_compute(ctx, x, weight, bias, activation, _SavedBuffer(act_inputs))

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def _forward_compute(
        ctx: FunctionCtx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation: str,
        saved: _SavedBuffer,
    ) -> torch.Tensor:
        """
        Compute e = activation(x @ weight.T + bias).
        This wrapper kicks the `kernel_fma` Triton kernel
        :param ctx: context for autograd
        :param x: input tensor
        :param weight: weight matrix
        :param bias: an optional bias tensor
        :param activation: Activation name. Needs to be a Triton kernel.
        :param act_inputs: an optional tensor to save the activation inputs (for backward)
        :return: result tensor
        """
        caller_saved = saved.tensor
        # Preserve original FP16 internal stores when AMP casts computation;
        # copy that stored value back to the actual caller-owned FP32 tensor.
        act_inputs = caller_saved
        if caller_saved is not None and caller_saved.dtype != x.dtype:
            act_inputs = torch.empty(caller_saved.shape, device=x.device, dtype=x.dtype)
        x_ = x if x.ndim == 2 else x.flatten(0, 1)

        assert x.dtype == weight.dtype, f"Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}"
        if bias is not None:
            assert x.dtype == bias.dtype, f"Input and bias must have the same dtype, got {x.dtype} and {bias.dtype}"
        assert x_.shape[1] == weight.shape[1], f"Incompatible dimensions: {x_.shape} - {weight.shape}"

        assert bias is None or bias.is_contiguous()
        assert bias is None or bias.shape[0] == weight.shape[0], "Incompatible dimensions in between weight and bias"
        assert weight.is_contiguous()

        M, K = x_.shape
        N, K = weight.shape

        outputs = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)  # noqa

        if M and N:
            with torch.cuda.device(x.device):
                kernel_fma_shifted[grid](
                    outputs,
                    act_inputs,
                    x_,
                    weight,  # data ptrs
                    bias if bias is not None else x,  # auto skip bias if not present
                    M,  # shapes
                    N,
                    K,
                    M // 32,  # key for triton cache (limit number of compilations)
                    N // 32,
                    K // 32,
                    output_m_stride=outputs.stride(0),  # strides
                    output_n_stride=outputs.stride(1),
                    act_inputs_m_stride=act_inputs.stride(0) if act_inputs is not None else 0,
                    act_inputs_n_stride=act_inputs.stride(1) if act_inputs is not None else 0,
                    a_m_stride=x_.stride(0),
                    a_k_stride=x_.stride(1),
                    b_n_stride=weight.stride(0),
                    b_k_stride=weight.stride(1),
                    HAS_BIAS=bias is not None,  # optional fused bias
                    SHOULD_SAVE_ACT_INPUTS=act_inputs is not None,  # optional save activation inputs
                    ACTIVATION=activation,  # optional fused activation
                    GROUP_M=8,  # speed optimization: group the programs
                )

        if caller_saved is not None and act_inputs is not caller_saved:
            caller_saved.copy_(act_inputs)

        outputs = outputs if x.ndim == 2 else outputs.reshape(x.shape[0], x.shape[1], N)
        ctx.save_for_backward(weight, bias, x)
        return outputs


def linear_layer_shifted(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation="",
    act_inputs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return LinearLayer.apply(x, weight, bias, activation, act_inputs)
