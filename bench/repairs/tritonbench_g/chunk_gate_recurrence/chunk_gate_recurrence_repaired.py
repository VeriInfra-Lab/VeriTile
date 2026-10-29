import torch
import triton
import triton.language as tl

torch.backends.cudnn.allow_tf32 = True

@triton.jit
def _fwd_recurrence(
    S, d, 
    O,
    NUM_HEAD, NUM_BLOCK, 
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr, BLOCK_MODEL_V: tl.constexpr,
    last_kv: tl.tensor,
    SEED_B: tl.constexpr, SEED_H: tl.constexpr,
    SEED_K: tl.constexpr, SEED_V: tl.constexpr
):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)    

    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]
    O = O + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K  +  tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]

    if last_kv is not None:
        last_kv = last_kv + (offset_bh // NUM_HEAD) * SEED_B + (offset_bh % NUM_HEAD) * SEED_H + (offset_d * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None]) * SEED_K + (offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]) * SEED_V
        acc = tl.load(last_kv).to(tl.float32)
    else:
        acc = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)

    tl.store(O, acc.to(O.dtype.element_ty))
    O += D_MODEL_K * D_MODEL_V
    d = d + offset_bh * NUM_BLOCK
    for i in range(NUM_BLOCK-1):
        d_i = tl.load(d)
        S_i = tl.load(S) 
        acc = acc * d_i + S_i
        tl.store(O, acc.to(O.dtype.element_ty))
        d += 1
        S += D_MODEL_K * D_MODEL_V
        O += D_MODEL_K * D_MODEL_V
     

## NUM_SPLIT_K/V. K/V dimension split into NUM_SPLIT_K/V parts with equal size BLOCK_MODEL
@triton.jit
def _bwd_recurrence(
    S, d, 
    DI, DG, DL, DS, 
    NUM_HEAD, NUM_BLOCK,
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr, BLOCK_MODEL_V: tl.constexpr,
    
 ):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)    

    # offset_h = offset_bh % NUM_HEAD
    NUM_K = D_MODEL_K // BLOCK_MODEL_K
    NUM_V = D_MODEL_V // BLOCK_MODEL_V
    # skip the last chunk because it is never used
    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    DI = DI + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    # start from the last chunk  
    DS = DS + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K  +  tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + (NUM_BLOCK - 1) * D_MODEL_K * D_MODEL_V

    DG = DG + offset_bh * NUM_BLOCK * NUM_K * NUM_V + offset_d * NUM_V + offset_s + (NUM_BLOCK - 2) * NUM_K * NUM_V

    d = d + offset_bh * NUM_BLOCK + (NUM_BLOCK - 1)

    Dacc = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32) 

    # ignore the first chunk
    for i in range(NUM_BLOCK - 1):
        S_i = tl.load(S)
        DS_i = tl.load(DS)
        d_i = tl.load(d)
        Dacc = Dacc * d_i + DS_i
        DG_i = tl.sum(Dacc * S_i.to(tl.float32))

        tl.store(DG, DG_i.to(DG.dtype.element_ty))
        tl.store(DI, Dacc.to(DI.dtype.element_ty))    

        S -= D_MODEL_K * D_MODEL_V
        DI -= D_MODEL_K * D_MODEL_V 
        DS -= D_MODEL_K * D_MODEL_V
        DG -= NUM_K * NUM_V
        d -= 1
    
    DL = DL + offset_bh * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K  +  tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]
    DS_i = tl.load(DS)
    d_i = tl.load(d)
    Dacc = Dacc * d_i + DS_i
    tl.store(DL, Dacc.to(DL.dtype.element_ty))  


def _span(tensor):
    if tensor.numel() == 0:
        return 0
    storage = tensor.untyped_storage()
    start, size = storage.data_ptr(), storage.nbytes()
    pointer = tensor.data_ptr()
    extent = 1 + sum((n-1)*stride for n,stride in zip(tensor.shape,tensor.stride()))
    if (pointer != start+4*tensor.storage_offset() or pointer%4 or start < 0
            or pointer < start or pointer+4*extent > start+size or start+size > 2**64):
        raise ValueError("FP32 views need aligned contained backing storage")
    return extent


def _payload(tensor, shape, device):
    if (not isinstance(tensor,torch.Tensor) or tensor.layout != torch.strided
            or tensor.dtype != torch.float32 or not tensor.is_cuda or tensor.device != device
            or tuple(tensor.shape) != tuple(shape) or tensor.is_neg() or tensor.is_conj()
            or any(stride < 0 for stride in tensor.stride())):
        raise ValueError("ordinary same-device FP32 CUDA tensors with the expected shape are required")
    _span(tensor)


def _inputs(kv, decay, seed):
    if not isinstance(kv,torch.Tensor) or kv.ndim != 5:
        raise ValueError("kv must have shape [B,H,N,K,V]")
    b,h,n,k,v = kv.shape
    if min(b,h,k,v) <= 0 or k%64 or v%16:
        raise ValueError("positive batch/head dimensions and complete 64x16 feature tiles are required")
    if n and (b*h >= 2**31 or k//64 > 65535 or v//16 > 65535):
        raise ValueError("the original three-dimensional CUDA grid must fit")
    _payload(kv,(b,h,n,k,v),kv.device)
    _payload(decay,(b,h,n),kv.device)
    if seed is not None:
        _payload(seed,(b,h,k,v),kv.device)


class _Range:
    def __init__(self,lo,hi=None,valid=True):
        self.lo,self.hi=lo,lo if hi is None else hi
        self.valid=valid and -(2**31)<=self.lo<=self.hi<2**31
    @staticmethod
    def of(value):
        return value if isinstance(value,_Range) else _Range(value)
    def __add__(self,other):
        other=self.of(other)
        return _Range(self.lo+other.lo,self.hi+other.hi,self.valid and other.valid)
    __radd__=__add__
    def __sub__(self,other):
        other=self.of(other)
        return _Range(self.lo-other.hi,self.hi-other.lo,self.valid and other.valid)
    def __rsub__(self,other):return self.of(other)-self
    def __mul__(self,other):
        other=self.of(other)
        corners=[a*b for a in (self.lo,self.hi) for b in (other.lo,other.hi)]
        return _Range(min(corners),max(corners),self.valid and other.valid)
    __rmul__=__mul__
    def __floordiv__(self,other):
        other=self.of(other)
        if self.lo<0 or other.lo<=0:return _Range(0,0,False)
        return _Range(self.lo//other.hi,self.hi//other.lo,self.valid and other.valid)
    def __mod__(self,other):
        other=self.of(other)
        if self.lo<0 or other.lo<=0:return _Range(0,0,False)
        return _Range(0,other.hi-1,self.valid and other.valid)


def _source_ranges(shape,seed_strides):
    b,h,n,k,v=shape
    head=_Range(0,b*h-1);tk=_Range(0,k//64-1);tv=_Range(0,v//16-1)
    e=_Range(0,63);j=_Range(0,15);step=_Range(0,n-1)
    n,k,v,h,kt,vt=map(_Range,(n,k,v,h,k//64,v//16))
    sb,sh,sk,sv=map(_Range,seed_strides)
    plane=k*v
    feature=tk*v*64+e*v+tv*16+j
    main=head*n*k*v+feature
    tiles=(k//64)*(v//16)
    dg=head*n*tiles+(tk*(v//16)+tv)
    seed=(head//h)*sb+(head%h)*sh+(tk*64+e)*sk+(tv*16+j)*sv
    return [n,k,v,h,kt,vt,plane,main,seed,main+step*plane,main+(step+1)*plane,
        main+(n-2-step)*plane,main+(n-1-step)*plane,dg+(n-2-step)*tiles,
        head*n+step,head*n+(n-1-step),head*k*v+feature]


def _launch_guard(shape,seed_strides,seed,tensors):
    if shape[2] == 0:
        return
    device=next(iter(tensors.values())).device
    if torch.cuda.current_device() != device.index:
        raise ValueError("the original launch requires its CUDA device to be current")
    expressions=_source_ranges(shape,seed_strides)
    if not all(expression.valid for expression in expressions):
        raise ValueError("every source signed32 integer intermediate must fit")
    roots={'x':(7,9),'output':(7,10,11),'incoming':(12,),'dx':(11,),
        'tileGate':(13,),'gate':(14,15),'dseed':(16,),'seed':(8,) if seed else ()}
    b,h,n,k,v=shape
    expected_shapes={'x':shape,'output':shape,'incoming':shape,'dx':shape,
        'gate':(b,h,n),'dseed':(b,h,k,v),'seed':(b,h,k,v),'tileGate':(b*h,n,k//64,v//16)}
    for role,tensor in tensors.items():
        _payload(tensor,expected_shapes[role],device)
        if role != 'seed' and _span(tensor) != tensor.numel():
            raise ValueError("prepared tensors must have a dense logical extent")
        if tensor.numel() and role != 'seed' and not tensor.is_contiguous():
            raise ValueError("original prepared launch tensors must be contiguous")
        for index in roots[role]:
            expression=expressions[index]
            if not 0 <= tensor.data_ptr()+4*expression.lo <= tensor.data_ptr()+4*expression.hi < 2**64:
                raise ValueError("all formed source pointers, including dead reverse pointers, must fit")


class ChunkGateRecurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, cross_decay, last_kv=None):
        _inputs(kv,cross_decay,last_kv)
        cross_decay = cross_decay.contiguous()
        kv = kv.contiguous()

        B, H, N, D_k, D_v = kv.shape 
        output = torch.empty_like(kv)        
        BLOCK_MODEL_K = 64
        BLOCK_MODEL_V = 16
    
        assert D_k % BLOCK_MODEL_K == 0
        assert D_v % BLOCK_MODEL_V == 0

        grid = (B*H, D_k//BLOCK_MODEL_K, D_v//BLOCK_MODEL_V)
        ctx.grid = grid
        ctx.have_last_kv = last_kv is not None
        ctx.BLOCK_MODEL_K = BLOCK_MODEL_K
        ctx.BLOCK_MODEL_V = BLOCK_MODEL_V

        if N == 0:
            ctx.save_for_backward(output, cross_decay)
            return output

        seed_strides = last_kv.stride() if last_kv is not None else (0, 0, 0, 0)
        ctx.seed_strides = seed_strides
        prepared = {'x':kv,'gate':cross_decay,'output':output}
        if last_kv is not None:
            prepared['seed'] = last_kv
        _launch_guard(kv.shape,seed_strides,last_kv is not None,prepared)
        _fwd_recurrence[grid](
            kv,
            cross_decay,
            output,
            D_MODEL_K=D_k, D_MODEL_V=D_v,
            NUM_BLOCK=N, NUM_HEAD=H,
            BLOCK_MODEL_K=BLOCK_MODEL_K,
            BLOCK_MODEL_V=BLOCK_MODEL_V,
            last_kv=last_kv,
            SEED_B=seed_strides[0], SEED_H=seed_strides[1],
            SEED_K=seed_strides[2], SEED_V=seed_strides[3]
        )

        ctx.save_for_backward(output, cross_decay)        
        return output

    @staticmethod
    def backward(ctx, DO):
        saved_output,saved_decay = ctx.saved_tensors
        _payload(DO,saved_output.shape,saved_output.device)
        _payload(saved_output,saved_output.shape,saved_output.device)
        _payload(saved_decay,saved_output.shape[:3],saved_output.device)
        DO = DO.contiguous()

        output, cross_decay = ctx.saved_tensors 

        B, H, N, D_k, D_v = output.shape 
        
        if N == 0:
            return (torch.zeros_like(DO), torch.zeros_like(cross_decay),
                    torch.zeros((B, H, D_k, D_v), device=output.device, dtype=output.dtype)
                    if ctx.have_last_kv else None)

        BLOCK_MODEL_K = 64
        BLOCK_MODEL_V = 16

        grid = (B*H, D_k//BLOCK_MODEL_K, D_v//BLOCK_MODEL_V)

        DI = torch.empty_like(DO)
        DG = torch.empty(B*H, N, D_k//BLOCK_MODEL_K, D_v//BLOCK_MODEL_V, device=cross_decay.device, dtype=cross_decay.dtype)
        DL = torch.empty(B, H, D_k, D_v, device=output.device, dtype=output.dtype)
        _launch_guard(output.shape,ctx.seed_strides,ctx.have_last_kv,
            {'output':output,'gate':cross_decay,'incoming':DO,'dx':DI,'tileGate':DG,'dseed':DL})
        _bwd_recurrence[grid](
            output, cross_decay,
            DI, DG, DL, DO, 
            NUM_HEAD=H, NUM_BLOCK = N, 
            D_MODEL_K = D_k,
            D_MODEL_V = D_v, 
            BLOCK_MODEL_K=BLOCK_MODEL_K,
            BLOCK_MODEL_V=BLOCK_MODEL_V,
        )

        DI[:, :, -1] = 0
        DG[:, -1] = 0
        DG = DG.view(B, H, N, -1).sum(dim=-1)
        return DI, DG, DL if ctx.have_last_kv else None

chunk_gate_recurrent = ChunkGateRecurrent.apply
