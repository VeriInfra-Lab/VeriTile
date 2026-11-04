
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    BLOCK_MODEL: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_bh % h
    off_e = tl.program_id(1)
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = v_offset
    # channel offset
    e_offset = off_e * BLOCK_MODEL

    ##### get block ptr
    Q_block_ptr = Q + qk_offset + tl.arange(0, d)[None, :]
    K_trans_block_ptr = K + qk_offset + tl.arange(0, d)[:, None]
    V_block_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_block_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]

    ##### init diag decay(Lambda); q, k decay; kv
    # q, k decay
    off_block = tl.arange(
        0, BLOCK
    )  # Not bug, this is a bit different from algorithm 1, but is mathematically equivalent
    # diag decay
    index = off_block[:, None] - off_block[None, :]
    kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)

    ##### compute
    for i in range(NUM_BLOCK):
        # load
        q = tl.load(
            Q_block_ptr + off_block[:, None] * d, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)
        k_trans = tl.load(
            K_trans_block_ptr + off_block[None, :] * d,
            mask=off_block[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr + off_block[:, None] * e, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)

        # compute
        qk = tl.dot(q, k_trans)
        qk = tl.where(index >= 0, qk, 0)
        o_intra = tl.dot(qk, v)
        o_inter = tl.dot(q, kv)
        o = o_intra + o_inter

        # save and update
        tl.store(
            O_block_ptr + off_block[:, None] * e,
            o.to(O_block_ptr.dtype.element_ty),
            mask=off_block[:, None] < n,
        )
        kv += tl.dot(k_trans, v)
        off_block += BLOCK


@triton.jit
def _bwd_intra_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_bh % h
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = v_offset
    block_offset = off_block * BLOCK + tl.arange(0, BLOCK)

    ##### get block ptr
    Q_trans_block_ptr = (
        Q + qk_offset + block_offset[None, :] * d + tl.arange(0, d)[:, None]
    )
    K_block_ptr = K + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[None, :]
    V_trans_block_ptr = (
        V + v_offset + block_offset[None, :] * e + tl.arange(0, e)[:, None]
    )

    DQ_block_ptr = DQ + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[None, :]
    DK_trans_block_ptr = (
        DK + qk_offset + block_offset[None, :] * d + tl.arange(0, d)[:, None]
    )
    DV_block_ptr = DV + v_offset + block_offset[:, None] * e + tl.arange(0, e)[None, :]
    DO_block_ptr = DO + o_offset + block_offset[:, None] * e + tl.arange(0, e)[None, :]

    ##### init diag decay(Lambda)
    array = tl.arange(0, BLOCK).to(tl.float32)
    # diag
    index = array[:, None] - array[None, :]

    ##### load block
    k = tl.load(K_block_ptr, mask=block_offset[:, None] < n, other=0.0).to(tl.float32)
    v_trans = tl.load(V_trans_block_ptr, mask=block_offset[None, :] < n, other=0.0).to(
        tl.float32
    )
    do = tl.load(DO_block_ptr, mask=block_offset[:, None] < n, other=0.0).to(tl.float32)
    q_trans = tl.load(Q_trans_block_ptr, mask=block_offset[None, :] < n, other=0.0).to(
        tl.float32
    )

    ##### compute
    dqk = tl.dot(do, v_trans)
    dqk = tl.where(index >= 0, dqk, 0)
    dq_intra = tl.dot(dqk, k)

    dk_intra_trans = tl.dot(q_trans, dqk)

    qk_trans = tl.dot(k, q_trans)
    qk_trans = tl.where(index <= 0, qk_trans, 0)
    dv_intra = tl.dot(qk_trans, do)

    dq = dq_intra
    dk_trans = dk_intra_trans
    dv = dv_intra

    # save
    tl.store(
        DQ_block_ptr,
        dq.to(DQ_block_ptr.dtype.element_ty),
        mask=block_offset[:, None] < n,
    )
    tl.store(
        DK_trans_block_ptr,
        dk_trans.to(DK_trans_block_ptr.dtype.element_ty),
        mask=block_offset[None, :] < n,
    )
    tl.store(
        DV_block_ptr,
        dv.to(DV_block_ptr.dtype.element_ty),
        mask=block_offset[:, None] < n,
    )


@triton.jit
def _bwd_inter_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = v_offset

    ##### get block ptr
    DQ_block_ptr = (
        DQ + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    K_block_ptr = (
        K + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V + v_offset + tl.arange(0, CBLOCK)[None, :] * e + tl.arange(0, e)[:, None]
    )
    DO_block_ptr = (
        DO + o_offset + tl.arange(0, CBLOCK)[:, None] * e + tl.arange(0, e)[None, :]
    )
    # mask
    off_block1 = tl.arange(0, CBLOCK)
    off_block2 = tl.arange(0, CBLOCK)

    ##### init lambda; kv
    kv_trans = tl.zeros([e, d], dtype=tl.float32)

    ##### compute dq inter
    for i in range(NUM_BLOCK):
        # compute in subblock
        for j in range(NUM_CBLOCK):
            if i > 0:  # if not add this, may have bug
                do = tl.load(DO_block_ptr, mask=off_block1[:, None] < n, other=0.0).to(
                    tl.float32
                )
                dq_inter = tl.dot(do, kv_trans)
                dq = dq_inter + tl.load(
                    DQ_block_ptr, mask=off_block1[:, None] < n, other=0.0
                )
                tl.store(
                    DQ_block_ptr,
                    dq.to(DQ_block_ptr.dtype.element_ty),
                    mask=off_block1[:, None] < n,
                )

            DQ_block_ptr += CBLOCK * d
            DO_block_ptr += CBLOCK * e
            off_block1 += CBLOCK

        # update kv in subblock
        kv_trans_current = tl.zeros([e, d], dtype=tl.float32)
        for j in range(NUM_CBLOCK):
            v_trans = tl.load(
                V_trans_block_ptr, mask=off_block2[None, :] < n, other=0.0
            ).to(tl.float32)
            k = tl.load(K_block_ptr, mask=off_block2[:, None] < n, other=0.0).to(
                tl.float32
            )
            kv_trans_current += tl.dot(v_trans, k)

            K_block_ptr += CBLOCK * d
            V_trans_block_ptr += CBLOCK * e
            off_block2 += CBLOCK

        kv_trans += kv_trans_current

    ##### get block ptr
    m = NUM_BLOCK * BLOCK
    off_block1 = m + tl.arange(0, CBLOCK)
    off_block2 = m + tl.arange(0, CBLOCK)

    Q_trans_block_ptr = (
        Q
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    K_block_ptr = (
        K
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + m * e
        + tl.arange(0, CBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )

    DK_trans_block_ptr = (
        DK
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    DV_block_ptr = (
        DV
        + v_offset
        + m * e
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + m * e
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    ##### init dkv
    dkv = tl.zeros([d, e], dtype=tl.float32)

    ##### compute dk, dv inter
    for i in range(NUM_BLOCK - 1, -1, -1):
        # compute in subblock
        for j in range(NUM_CBLOCK - 1, -1, -1):
            K_block_ptr -= CBLOCK * d
            V_trans_block_ptr -= CBLOCK * e
            DK_trans_block_ptr -= CBLOCK * d
            DV_block_ptr -= CBLOCK * e
            off_block1 -= CBLOCK

            if i < NUM_BLOCK - 1:  # if not add this, may have bug
                k = tl.load(K_block_ptr, mask=off_block1[:, None] < n, other=0.0).to(
                    tl.float32
                )
                v_trans = tl.load(
                    V_trans_block_ptr, mask=off_block1[None, :] < n, other=0.0
                ).to(tl.float32)

                dk_inter_trans = tl.dot(dkv, v_trans)
                dv_inter = tl.dot(k, dkv)

                dk_trans = dk_inter_trans + tl.load(
                    DK_trans_block_ptr, mask=off_block1[None, :] < n, other=0.0
                )
                dv = dv_inter + tl.load(
                    DV_block_ptr, mask=off_block1[:, None] < n, other=0.0
                )

                tl.store(
                    DK_trans_block_ptr,
                    dk_trans.to(DK_trans_block_ptr.dtype.element_ty),
                    mask=off_block1[None, :] < n,
                )
                tl.store(
                    DV_block_ptr,
                    dv.to(DV_block_ptr.dtype.element_ty),
                    mask=off_block1[:, None] < n,
                )

        # update dkv in subblock
        dkv_current = tl.zeros([d, e], dtype=tl.float32)
        for j in range(NUM_CBLOCK - 1, -1, -1):
            DO_block_ptr -= CBLOCK * e
            Q_trans_block_ptr -= CBLOCK * d
            off_block2 -= CBLOCK

            do = tl.load(DO_block_ptr, mask=off_block2[:, None] < n, other=0.0).to(
                tl.float32
            )
            q_trans = tl.load(
                Q_trans_block_ptr, mask=off_block2[None, :] < n, other=0.0
            ).to(tl.float32)
            dkv_current += tl.dot(q_trans, do)

        dkv += dkv_current


def _check_pointers(tensors, formed_limit):
    for tensor in tensors:
        pointer = tensor.data_ptr()
        if pointer % 4 or pointer < 0 or pointer + 4 * formed_limit >= 2**63:
            raise ValueError("aligned FP32 pointers with signed-64 formation room required")


class LightningAttention2NoDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        if any(type(t) is not torch.Tensor for t in (q, k, v)):
            raise TypeError("ordinary torch.Tensor inputs required")
        if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
            raise ValueError("expected Q/K[B,H,N,D] and V[B,H,N,E]")
        b, h, n, d = q.shape
        e = v.shape[-1]
        if min(b, h, n) <= 0 or min(d, e) < 16 or d & (d - 1) or e & (e - 1):
            raise ValueError("positive B/H/N and power-of-two D/E >= 16 required")
        if any(t.layout != torch.strided or t.dtype != torch.float32 or
               not t.is_cuda or t.device != q.device for t in (q, k, v)):
            raise ValueError("same-device strided CUDA FP32 tensors required")
        if any(not t.is_contiguous() or t.is_neg() or t.is_conj() for t in (q, k, v)):
            raise ValueError("contiguous inputs with resolved logical values required")
        if max(triton.cdiv(n, 64), triton.cdiv(e, min(e, 32))) > 65535:
            raise ValueError("kernel launch y axis exceeds 65535")
        if (b * h * n + 128) * max(d, e) >= 2**31:
            raise ValueError("formed tensor offsets exceed signed-int32 contract")
        _check_pointers((q, k, v), (b * h * n + 128) * max(d, e))

        b, h, n, d = q.shape
        e = v.shape[-1]
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
        _check_pointers((o,), (b * h * n + 128) * max(d, e))

        BLOCK = 64
        NUM_BLOCK = triton.cdiv(q.shape[2], BLOCK)
        # parallel over channel
        BLOCK_MODEL = min(triton.next_power_of_2(e), 32)
        grid = (b * h, triton.cdiv(e, BLOCK_MODEL))

        with torch.cuda.device(q.device):
            _fwd_kernel[grid](
                q,
                k,
                v,
                o,
                b,
                h,
                n,
                d,
                e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                BLOCK_MODEL=BLOCK_MODEL,
                enable_fp_fusion=True,
            )

        ctx.save_for_backward(q, k, v)

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors

        if type(do) is not torch.Tensor:
            raise TypeError("ordinary incoming cotangent required")
        if (do.shape != v.shape or do.layout != torch.strided or do.dtype != torch.float32
                or not do.is_cuda or do.device != q.device or do.is_neg() or do.is_conj()):
            raise ValueError("same-shape/device resolved strided FP32 cotangent required")
        # Autograd sum() supplies a zero-stride cotangent; retain its required materialization.
        do = do.contiguous()
        _check_pointers((q, k, v, do), (q.shape[0] * q.shape[1] * q.shape[2] + 128)
                        * max(q.shape[3], v.shape[3]))

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _check_pointers((dq, dk, dv), (q.shape[0] * q.shape[1] * q.shape[2] + 128)
                        * max(q.shape[3], v.shape[3]))

        b, h, n, d = q.shape
        e = v.shape[-1]

        # block size
        BLOCK = 64
        NUM_BLOCK = triton.cdiv(n, BLOCK)
        # compute block size
        CBLOCK = 32
        NUM_CBLOCK = BLOCK // CBLOCK

        # for intra part, compute in parallel
        grid = (b * h, NUM_BLOCK)
        with torch.cuda.device(q.device):
            _bwd_intra_kernel[grid](
                q,
                k,
                v,
                do,
                dq,
                dk,
                dv,
                b,
                h,
                n,
                d,
                e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
                enable_fp_fusion=True,
            )

        # for inter part, compute in sequencial
        grid = (b * h,)
        with torch.cuda.device(q.device):
            _bwd_inter_kernel[grid](
                q,
                k,
                v,
                do,
                dq,
                dk,
                dv,
                b,
                h,
                n,
                d,
                e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
                enable_fp_fusion=True,
            )

        return dq, dk, dv


lightning_attn2_no_decay = LightningAttention2NoDecay.apply
