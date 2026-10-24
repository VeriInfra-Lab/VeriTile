
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["BT", "BK", "BV"], 
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k,
    v,
    d, 
    v_new,
    h,
    initial_state,
    final_state,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_w_b, s_w_h, s_w_t, s_w_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(initial_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_h_cumsum = tl.zeros([BK, BV], dtype=tl.float32)
        for i_c in range(tl.cdiv(BT, BC)):
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_d = tl.make_block_ptr(d + (i_bh // H) * s_w_b + (i_bh % H) * s_w_h, (T, K), (s_w_t, s_w_d), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
            p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            p_v_new = tl.make_block_ptr(v_new + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))   
            b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
            b_d = tl.load(p_d, boundary_check=(0, 1), padding_option="zero")
            b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
            b_v -= tl.dot(b_d, b_h.to(b_k.dtype), allow_tf32=False)
            tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))
            b_h_cumsum += tl.dot(b_k, b_v.to(b_k.dtype), allow_tf32=False)
        b_h += b_h_cumsum      
        
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(final_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

def _tensor(tensor, shape, name, device=None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if (tensor.layout != torch.strided or tensor.dtype != torch.float32 or
            not tensor.is_cuda or tensor.is_neg() or tensor.is_conj() or
            not tensor.is_contiguous()):
        raise ValueError(f"{name} must be an ordinary contiguous FP32 CUDA tensor")
    if tuple(tensor.shape) != tuple(shape) or (device is not None and tensor.device != device):
        raise ValueError(f"{name} has the wrong shape or device")


def _pointer(tensor, extent):
    address = tensor.data_ptr()
    if address % 4 or address < 0 or address + 4 * extent + 4 >= 2**63:
        raise ValueError("formed FP32 pointers must be aligned and fit signed 64-bit bytes")


def _formed(B, H, T, K, V, BT, BC, BK, BV, NT):
    time = NT * BT + triton.cdiv(BT, BC) * BC
    value = triton.cdiv(V, BV) * BV
    key = B * H * T * K + BK + time * K
    weight = B * (H*T*K) + H * (T*K) + time * K + BK
    values = B * H * T * V + time * V + value
    history = B * H * NT * K * V + NT * K * V + BK * V + value
    state = B * H * K * V + BK * V + value
    return key + weight + values + history + state + 1


def chunk_fwd_h_fn_static_dot(k, w, u, BT, initial_state=None, final_state=None):
    """Separate static-dot contract candidate; the kernel body is unchanged."""
    if not isinstance(k, torch.Tensor) or not isinstance(u, torch.Tensor):
        raise TypeError("K and U must be tensors")
    if k.ndim != 4 or u.ndim != 4:
        raise ValueError("expected K/W[B,H,T,K] and U[B,H,T,V]")
    B, H, T, K = k.shape
    V = u.shape[-1]
    if type(BT) is not int or BT <= 0:
        raise ValueError("BT must be a positive builtin integer")
    if not 1 <= K <= 256:
        raise ValueError("the source requires 1 <= K <= 256")
    _tensor(k, (B,H,T,K), "K")
    _tensor(w, (B,H,T,K), "W", k.device)
    _tensor(u, (B,H,T,V), "U", k.device)
    for name, tensor in [("initial_state", initial_state), ("final_state", final_state)]:
        if tensor is not None:
            _tensor(tensor, (B,H,K,V), name, k.device)
    BK = triton.next_power_of_2(K)
    BV = 16 if BK > 128 else 32 if BK > 64 else 64
    BC = min(BT, BV)
    if BC & (BC - 1):
        raise ValueError("the selected BC must be a power of two")
    if B * H * V and (BK < 16 or BC < 16):
        raise ValueError("launched FP32 dots require BK >= 16 and BC >= 16")
    NT, NV = triton.cdiv(T, BT), triton.cdiv(V, BV)
    if B*H > 65535 or NV > 65535:
        raise ValueError("the source grid exceeds CUDA y/z limits")
    extent = _formed(B,H,T,K,V,BT,BC,BK,BV,NT)
    # Every base/loop intermediate is nonnegative and below this checked sum.
    # Block-pointer products are signed 64-bit; source scalar arithmetic stays i32.
    if max(B,H,T,K,V,BT,NT,NV,H*T*K,T*K,T*V,NT*K*V,extent) >= 2**31:
        raise ValueError("formed source offsets exceed signed int32")
    tensors = [k,w,u] + [t for t in (initial_state,final_state) if t is not None]
    for tensor in tensors:
        _pointer(tensor, extent)
    if final_state is not None and final_state.numel():
        low = final_state.data_ptr()
        high = low + 4 * final_state.numel()
        for tensor in [k,w,u] + ([] if initial_state is None else [initial_state]):
            if tensor.numel() and low < tensor.data_ptr() + 4*tensor.numel() and tensor.data_ptr() < high:
                raise ValueError("final state must be separated from read inputs")
    with torch.cuda.device(k.device):
        h = torch.empty((B,H,NT*K,V), dtype=torch.float32, device=k.device)
        v_new = torch.empty((B,H,T,V), dtype=torch.float32, device=k.device)
        _pointer(h, extent)
        _pointer(v_new, extent)
        if B*H*V:
            chunk_delta_rule_fwd_kernel_h[(1,NV,B*H)](
                k,u,w,v_new,h,initial_state,final_state,
                T*K,K,1,H*T*K,T*K,K,1,T*V,V,1,NT*K*V,V,
                H=H,T=T,K=K,V=V,BT=BT,BC=BC,BK=BK,BV=BV,NT=NT,
                USE_INITIAL_STATE=initial_state is not None,
                STORE_FINAL_STATE=final_state is not None)
        return h,v_new


chunk_fwd_h_fn = chunk_fwd_h_fn_static_dot
