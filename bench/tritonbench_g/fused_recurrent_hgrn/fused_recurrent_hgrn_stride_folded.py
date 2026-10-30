
import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.autotune(
    configs=[
        triton.Config({'BD': 32}, num_warps=1),
        triton.Config({'BD': 32}, num_warps=2),
        triton.Config({'BD': 32}, num_warps=4),
        triton.Config({'BD': 32}, num_warps=8),
        triton.Config({'BD': 64}, num_warps=1),
        triton.Config({'BD': 64}, num_warps=2),
        triton.Config({'BD': 64}, num_warps=4),
        triton.Config({'BD': 64}, num_warps=8),
        triton.Config({'BD': 128}, num_warps=1),
        triton.Config({'BD': 128}, num_warps=2),
        triton.Config({'BD': 128}, num_warps=4),
        triton.Config({'BD': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def fused_recurrent_hgrn_fwd_kernel(
    x,
    g,
    o,
    h0,
    ht,
    T: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_x = x + i_bh * (T * D) + o_d
    p_g = g + i_bh * (T * D) + o_d
    p_o = o + i_bh * (T * D) + o_d

    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * D + o_d
        b_h += tl.load(p_h0, mask=mask, other=0).to(tl.float32)
    for _ in range(0, T):
        b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_h = b_g * b_h + b_x
        tl.store(p_o, b_h.to(p_o.dtype.element_ty), mask=mask)

        p_x += D
        p_g += D
        p_o += D

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * D + o_d
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BD': 32}, num_warps=1),
        triton.Config({'BD': 32}, num_warps=2),
        triton.Config({'BD': 32}, num_warps=4),
        triton.Config({'BD': 32}, num_warps=8),
        triton.Config({'BD': 64}, num_warps=1),
        triton.Config({'BD': 64}, num_warps=2),
        triton.Config({'BD': 64}, num_warps=4),
        triton.Config({'BD': 64}, num_warps=8),
        triton.Config({'BD': 128}, num_warps=1),
        triton.Config({'BD': 128}, num_warps=2),
        triton.Config({'BD': 128}, num_warps=4),
        triton.Config({'BD': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def fused_recurrent_hgrn_bwd_kernel(
    g,
    o,
    dx,
    dg,
    do,
    h0,
    dht,
    T: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_GRAD: tl.constexpr
):
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    b_dh = tl.zeros([BD], dtype=tl.float32)
    if USE_FINAL_GRAD:
        b_dh = tl.load(dht + i_bh * D + o_d, mask=mask, other=0).to(tl.float32)
    for step in range(0, T):
        i = T - 1 - step
        p_g = g + (i_bh * T + i) * D + o_d
        p_dx = dx + (i_bh * T + i) * D + o_d
        p_dg = dg + (i_bh * T + i) * D + o_d
        p_do = do + (i_bh * T + i) * D + o_d
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)
        if i > 0:
            b_o = tl.load(o + (i_bh * T + i - 1) * D + o_d, mask=mask, other=0).to(tl.float32)
        elif USE_INITIAL_STATE:
            b_o = tl.load(h0 + i_bh * D + o_d, mask=mask, other=0).to(tl.float32)
        else:
            b_o = tl.zeros([BD], dtype=tl.float32)

        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dg = b_dh * b_o
        b_dh = b_dh * b_g
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=mask)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=mask)




def _payload(tensor,shape,device):
    if (not isinstance(tensor,torch.Tensor) or tensor.layout != torch.strided
            or tensor.dtype != torch.float32 or not tensor.is_cuda or tensor.device != device
            or tuple(tensor.shape) != tuple(shape) or not tensor.is_contiguous()
            or tensor.is_neg() or tensor.is_conj() or any(s<0 for s in tensor.stride())):
        raise ValueError("ordinary contiguous same-device FP32 CUDA tensors are required")
    if tensor.numel():
        storage=tensor.untyped_storage()
        base,size=storage.data_ptr(),storage.nbytes()
        pointer=tensor.data_ptr()
        if (pointer != base+4*tensor.storage_offset() or pointer%4 or pointer<base
                or pointer+4*tensor.numel()>base+size or base+size>2**64):
            raise ValueError("FP32 tensors need aligned contained backing storage")


def _forward_inputs(x,g,seed,final):
    if not isinstance(x,torch.Tensor) or x.ndim != 4:
        raise ValueError("x must have shape [B,H,T,D]")
    b,h,t,d=x.shape
    if min(b,h,d)<=0 or not isinstance(final,bool):
        raise ValueError("positive B/H/D and a Boolean final-state flag are required")
    _payload(x,x.shape,x.device)
    _payload(g,x.shape,x.device)
    if seed is not None:_payload(seed,(b,h,d),x.device)


def _launch_guard(shape,tensors):
    b,h,t,d=shape
    # These cover every original autotune BD and every loop endpoint.
    high=b*h*(t+1)*d+d+127
    if b*h>65535 or (d+31)//32>=2**31 or high>=2**31:
        raise ValueError("all original CUDA grids and signed32 source intermediates must fit")
    active=[tensor for tensor in tensors if tensor is not None]
    device=active[0].device
    if torch.cuda.current_device()!=device.index:
        raise ValueError("the original launch requires its CUDA device to be current")
    for tensor in active:
        _payload(tensor,tensor.shape,device)
        if tensor.data_ptr()+4*high>=2**64:
            raise ValueError("every formed pointer including padded and final lanes must fit")


class FusedRecurrentHGRNFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, g, initial_state=None, output_final_state=False):
        _forward_inputs(x,g,initial_state,output_final_state)
        B, H, T, D = x.shape

        final_state = None
        if output_final_state:
            final_state = x.new_empty(B, H, D)

        o = torch.empty_like(x)
        def grid(meta): return (triton.cdiv(D, meta['BD']), B * H)
        _launch_guard(x.shape,[x,g,o,initial_state,final_state])
        fused_recurrent_hgrn_fwd_kernel[grid](
            x, g, o, initial_state, final_state,
            T, D,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None
        )
        ctx.set_materialize_grads(False)
        ctx.output_final_state = output_final_state
        ctx.save_for_backward(g, o, initial_state)
        return o, final_state

    @staticmethod
    def backward(ctx, do, dht=None):
        g, o, initial_state = ctx.saved_tensors
        B, H, T, D = o.shape
        if do is not None:
            _payload(do,o.shape,o.device)
        if ctx.output_final_state and dht is not None:
            _payload(dht,(B,H,D),o.device)
        do = torch.zeros_like(o) if do is None else do

        dx = torch.empty_like(o)
        dg = torch.empty_like(g)
        def grid(meta): return (triton.cdiv(D, meta['BD']), B * H)
        _launch_guard(o.shape,[g,o,dx,dg,do,initial_state,
            dht if ctx.output_final_state else None])
        fused_recurrent_hgrn_bwd_kernel[grid](
            g, o, dx, dg, do, initial_state, dht,
            T, D,
            USE_INITIAL_STATE=initial_state is not None,
            USE_FINAL_GRAD=ctx.output_final_state and dht is not None,
        )

        return dx, dg, None, None


def fused_recurrent_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = FusedRecurrentHGRNFunction.apply(x, g, initial_state, output_final_state)
    return o, final_state




