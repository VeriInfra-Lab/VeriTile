import VeriTile.Triton.DSL.PythonBlockAdvances
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
/-! Complete repaired Python bodies. DQ, saved maxima and the scalar scratch
use the host's fixed FP32 storage. The backward call uses BLOCK_N=BLOCK_M,
matching the actual launch; source names, casts and statements are retained. -/
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 12000000
def forwardTypedSignature (dtype out_dtype : FloatDType) : CallSignature where
  floatingPointers := [("Q",dtype), ("K",dtype), ("V",dtype), ("L",out_dtype), ("M",out_dtype), ("Out",dtype)]
  scalars := [("sm_scale",.fp32), ("stride_qz",.integer), ("stride_qh",.integer), ("stride_qm",.integer), ("stride_qk",.integer), ("stride_kz",.integer), ("stride_kh",.integer), ("stride_kn",.integer), ("stride_kk",.integer), ("stride_vz",.integer), ("stride_vh",.integer), ("stride_vk",.integer), ("stride_vn",.integer), ("stride_oz",.integer), ("stride_oh",.integer), ("stride_om",.integer), ("stride_on",.integer), ("Z",.integer), ("H",.integer), ("N_CTX",.integer), ("D0",.integer)]
  constexpr := ["BLOCK_M", "BLOCK_DMODEL", "BLOCK_N"]
  readPorts := ["Q", "K", "V"]
  writePorts := ["L", "M", "Out"]

def forwardTypedKernel (dtype out_dtype : FloatDType)
    (Q K V L M Out : RegionName)
    (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel :=
  with_python_block_advances in
  with_triton_call (forwardTypedSignature dtype out_dtype) call_args [sm_scale:=sm_scale, stride_qz:=stride_qz, stride_qh:=stride_qh, stride_qm:=stride_qm, stride_qk:=stride_qk, stride_kz:=stride_kz, stride_kh:=stride_kh, stride_kn:=stride_kn, stride_kk:=stride_kk, stride_vz:=stride_vz, stride_vh:=stride_vh, stride_vk:=stride_vk, stride_vn:=stride_vn, stride_oz:=stride_oz, stride_oh:=stride_oh, stride_om:=stride_om, stride_on:=stride_on, Z:=Z, H:=H, N_CTX:=N_CTX, D0:=D0] in triton {
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # initialize offsets for store
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    stride_qh_2d = stride_qh // stride_qm // stride_qk

    q_tile_ptr = tl.make_block_ptr(base=Q,
                                   shape=(D0, BLOCK_DMODEL),
                                   strides=(stride_qm, stride_qk),
                                   offsets=(
                                       off_hz * stride_qh_2d + start_m * BLOCK_M, 0),
                                   block_shape=(BLOCK_M, BLOCK_DMODEL),
                                   order=(1, 0))
    k_tile_ptr = tl.make_block_ptr(base=K,
                                   shape=(D0, BLOCK_DMODEL),
                                   strides=(stride_kn, stride_kk),
                                   offsets=(off_hz * stride_qh_2d, 0),
                                   block_shape=(BLOCK_N, BLOCK_DMODEL),
                                   order=(1, 0))
    v_tile_ptr = tl.make_block_ptr(base=V,
                                   shape=(D0, BLOCK_DMODEL),
                                   strides=(stride_vk, stride_vn),
                                   offsets=(off_hz * stride_qh_2d, 0),
                                   block_shape=(BLOCK_N, BLOCK_DMODEL),
                                   order=(1, 0))
    out_tile_ptr = tl.make_block_ptr(base=Out,
                                     shape=(D0, BLOCK_DMODEL),
                                     strides=(stride_om, stride_on),
                                     offsets=(off_hz * stride_qh_2d + start_m * BLOCK_M, 0),
                                     block_shape=(BLOCK_M, BLOCK_DMODEL),
                                     order=(1, 0))
    # load q: it will stay in SRAM throughout
    q = tl.load(q_tile_ptr)

    # loop over k, v and update accumulators
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        # -- compute qk ----
        k = tl.load(k_tile_ptr, boundary_check=(0, 1))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (
            start_n + offs_n[None, :]), qk, float("-inf"))
        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        # correct old l
        l_prev *= tl.exp(m_prev - m_curr)
        # attention weights
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        # rescale operands of matmuls
        l_rcp = 1. / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(tl.float16)
        v = tl.load(v_tile_ptr, boundary_check=(0, 1))
        acc += tl.dot(p.to(v.dtype), v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_tile_ptr = tl.advance(k_tile_ptr, [BLOCK_N, 0])
        v_tile_ptr = tl.advance(v_tile_ptr, [BLOCK_N, 0])
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_prev)
    tl.store(m_ptrs, m_prev)

    acc = acc.to(tl.float16)
    tl.store(out_tile_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))
  }

def preprocessTypedSignature (dtype out_dtype : FloatDType) : CallSignature where
  floatingPointers := [("Out",dtype), ("DO",dtype), ("L",out_dtype), ("NewDO",dtype), ("Delta",out_dtype)]
  scalars := []
  constexpr := ["BLOCK_M", "D_HEAD"]
  readPorts := ["Out", "DO", "L"]
  writePorts := ["NewDO", "Delta"]

def preprocessTypedKernel (dtype out_dtype : FloatDType)
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat) :
    ComputeKernel :=
  with_python_block_advances in
  with_triton_call (preprocessTypedSignature dtype out_dtype) call_args [] in triton {
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    denom = tl.load(L + off_m).to(tl.float32)
    # compute
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
    tl.store(Delta + off_m, delta)
  }

def backwardTypedSignature (dtype : FloatDType) : CallSignature where
  floatingPointers := [("DQ",.fp32),("L",.fp32),("M",.fp32),("D",.fp32),("Q",dtype),("K",dtype),("V",dtype),("Out",dtype),("DO",dtype),("DK",dtype),("DV",dtype)]
  scalars := [("sm_scale",.fp32), ("stride_qz",.integer), ("stride_qh",.integer), ("stride_qm",.integer), ("stride_qk",.integer), ("stride_kz",.integer), ("stride_kh",.integer), ("stride_kn",.integer), ("stride_kk",.integer), ("stride_vz",.integer), ("stride_vh",.integer), ("stride_vk",.integer), ("stride_vn",.integer), ("Z",.integer), ("H",.integer), ("N_CTX",.integer), ("D0",.integer), ("num_block",.integer)]
  constexpr := ["BLOCK_M", "BLOCK_DMODEL", "BLOCK_N"]
  readPorts := ["Q", "K", "V", "DO", "M", "D", "DQ"]
  writePorts := ["DQ", "DV", "DK"]

def backwardTypedKernel (dtype : FloatDType)
    (Q K V Out DO DQ DK DV L M D : RegionName)
    (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      Z H N_CTX D0 num_block BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel :=
  let BLOCK_N := BLOCK_M
  with_python_block_advances in
  with_triton_call (backwardTypedSignature dtype) call_args [sm_scale:=sm_scale, stride_qz:=stride_qz, stride_qh:=stride_qh, stride_qm:=stride_qm, stride_qk:=stride_qk, stride_kz:=stride_kz, stride_kh:=stride_kh, stride_kn:=stride_kn, stride_kk:=stride_kk, stride_vz:=stride_vz, stride_vh:=stride_vh, stride_vk:=stride_vk, stride_vn:=stride_vn, Z:=Z, H:=H, N_CTX:=N_CTX, D0:=D0, num_block:=num_block] in triton {
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    # init tile_ptr
    stride_qz_2d = stride_qz // stride_qm // stride_qk
    stride_qh_2d = stride_qh // stride_qm // stride_qk

    q_tile_ptr = tl.make_block_ptr(base=Q,
                                   shape=(D0, BLOCK_DMODEL),
                                   strides=(stride_qm, stride_qk),
                                   offsets=(
                                       off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
                                   block_shape=(BLOCK_M, BLOCK_DMODEL),
                                   order=(1, 0))
    k_tile_ptr = tl.make_block_ptr(base=K,
                                   shape=(D0, BLOCK_DMODEL),
                                   strides=(stride_kn, stride_kk),
                                   offsets=(
                                       off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
                                   block_shape=(BLOCK_M, BLOCK_DMODEL),
                                   order=(1, 0))
    v_tile_ptr = tl.make_block_ptr(base=V,
                                   shape=(D0, BLOCK_DMODEL),
                                   strides=(stride_vk, stride_vn),
                                   offsets=(
                                       off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
                                   block_shape=(BLOCK_M, BLOCK_DMODEL),
                                   order=(1, 0))
    do_tile_ptr = tl.make_block_ptr(base=DO,
                                    shape=(D0, BLOCK_DMODEL),
                                    strides=(stride_qm, stride_qk),
                                    offsets=(
                                        off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
                                    block_shape=(BLOCK_M, BLOCK_DMODEL),
                                    order=(1, 0))
    dq_tile_ptr = tl.make_block_ptr(base=DQ,
                                    shape=(D0, BLOCK_DMODEL),
                                    strides=(stride_qm, stride_qk),
                                    offsets=(
                                        off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
                                    block_shape=(BLOCK_M, BLOCK_DMODEL),
                                    order=(1, 0))
    dk_tile_ptr = tl.make_block_ptr(base=DK,
                                    shape=(D0, BLOCK_DMODEL),
                                    strides=(stride_qm, stride_qk),
                                    offsets=(
                                        off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
                                    block_shape=(BLOCK_M, BLOCK_DMODEL),
                                    order=(1, 0))
    dv_tile_ptr = tl.make_block_ptr(base=DV,
                                    shape=(D0, BLOCK_DMODEL),
                                    strides=(stride_qm, stride_qk),
                                    offsets=(
                                        off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
                                    block_shape=(BLOCK_M, BLOCK_DMODEL),
                                    order=(1, 0))
    # offset pointers for batch/head
    DQ += off_z * stride_qz + off_h * stride_qh
    for start_n in range(0, num_block):
        lo = start_n * BLOCK_M
        # initialize row/col offsets
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        m_ptrs = M + off_hz * N_CTX
        # initialize dv amd dk
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        k = tl.load(k_tile_ptr, boundary_check=(0, 1))
        v = tl.load(v_tile_ptr, boundary_check=(0, 1))
        # loop over rows
        for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            q = tl.load(q_tile_ptr, boundary_check=(0, 1))
            # recompute p = softmax(qk, dim=-1).T
            # NOTE: `do` is pre-divided by `l`; no normalization here
            qk = tl.dot(q, tl.trans(k))
            qk = tl.where(offs_m_curr[:, None] >= (
                offs_n[None, :]), qk, float("-inf"))
            m = tl.load(m_ptrs + offs_m_curr)
            p = tl.exp(qk * sm_scale - m[:, None])
            # compute dv
            do = tl.load(do_tile_ptr, boundary_check=(0, 1))
            dv += tl.dot(tl.trans(p.to(tl.float16).to(do.dtype)), do)
            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, tl.trans(v))
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds.to(tl.float16).to(q.dtype)), q)
            # compute dq
            dq = tl.load(dq_tile_ptr)
            dq += tl.dot(ds.to(tl.float16).to(k.dtype), k)
            tl.store(dq_tile_ptr, dq)
            # increment pointers
            dq_ptrs += BLOCK_M * stride_qm
            q_tile_ptr = tl.advance(q_tile_ptr, [BLOCK_M, 0])
            do_tile_ptr = tl.advance(do_tile_ptr, [BLOCK_M, 0])
            dq_tile_ptr = tl.advance(dq_tile_ptr, [BLOCK_M, 0])
        q_tile_ptr = tl.advance(q_tile_ptr, [lo + (1 - num_block) * BLOCK_M, 0])
        do_tile_ptr = tl.advance(do_tile_ptr, [lo + (1 - num_block) * BLOCK_M, 0])
        dq_tile_ptr = tl.advance(dq_tile_ptr, [lo + (1 - num_block) * BLOCK_M, 0])
        # increment tile pointers
        k_tile_ptr = tl.advance(k_tile_ptr, [BLOCK_M, 0])
        v_tile_ptr = tl.advance(v_tile_ptr, [BLOCK_M, 0])
        # write-back
        tl.store(dv_tile_ptr, dv.to(tl.float16).to(DV.dtype.element_ty), boundary_check=(0, 1))
        tl.store(dk_tile_ptr, dk.to(tl.float16).to(DK.dtype.element_ty), boundary_check=(0, 1))
        dv_tile_ptr = tl.advance(dv_tile_ptr, [BLOCK_M, 0])
        dk_tile_ptr = tl.advance(dk_tile_ptr, [BLOCK_M, 0])
  }

def preprocessShiftedTypedKernel (dtype out_dtype : FloatDType)
    (Out DO L NewDO Delta : RegionName) (D_HEAD : Nat) :
    ComputeKernel :=
  let BLOCK_M : Nat := 128
  with_python_block_advances in
  with_triton_call (preprocessTypedSignature dtype out_dtype) call_args [] in triton {
    tl.static_assert(BLOCK_M == 128)
    off_m = (tl.program_id(0) << 7) + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    denom = tl.load(L + off_m).to(tl.float32)
    # compute
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
    tl.store(Delta + off_m, delta)
  }

end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
