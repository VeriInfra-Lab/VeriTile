import VeriTile.Triton.DSL.PositionalLoadOther
namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

def halfSignature : CallSignature where
  pointers := [("Q",.fp16),("K",.fp16),("V",.fp16),("Out",.fp16)]
  scalars := [("sm_scale",.fp32),("stride_qz",.integer),("stride_qh",.integer),("stride_qm",.integer),("stride_qk",.integer),("stride_kz",.integer),("stride_kh",.integer),("stride_kn",.integer),("stride_kk",.integer),("stride_vz",.integer),("stride_vh",.integer),("stride_vk",.integer),("stride_vn",.integer),("stride_oz",.integer),("stride_oh",.integer),("stride_om",.integer),("stride_on",.integer),("N_HEAD",.integer),("H",.integer),("N_CTX",.integer),("start_position",.integer)]
  constexpr := ["IS_CAUSAL","BLOCK_M","BLOCK_N","BLOCK_DMODEL","USE_FP8"]
  readPorts := ["Q","K","V"]
  writePorts := ["Out"]

def halfKernel
    (Q K V Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel :=
  let USE_FP8 := Bool.false
  with_triton_call halfSignature call_args [stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_vz := stride_vz, stride_vh := stride_vh, stride_vk := stride_vk, stride_vn := stride_vn, stride_oz := stride_oz, stride_oh := stride_oh, stride_om := stride_om, stride_on := stride_on, N_HEAD := N_HEAD, H := H, N_CTX := N_CTX, start_position := start_position] in triton {
    start_m = tl.program_id(0)

    head_idx = tl.program_id(1)
    batch_id = head_idx // N_HEAD
    off_hz = head_idx % N_HEAD

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = batch_id * stride_qz + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk  # <- stride fixed
    off_k = batch_id * stride_kz + off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk  # <- stride fixed
    off_v = batch_id * stride_vz + off_hz * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn  # <- stride fixed
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs, offs_m[:, None] < H, other=0.0)
    # loop over k, v and update accumulator
    block_n_end = N_CTX  # <- ADDED (including the IF)
    if IS_CAUSAL:
        # in causal mode, we expect that BLOCK_M_SIZE == BLOCK_N_SIZE
        # autotune will prune shapes not matching this rule
        block_n_end = (start_m + 1) * BLOCK_N + start_position
    for start_n in range(0, block_n_end, BLOCK_N):
        block_n_offs = start_n + offs_n  # <- ADDED
        # -- compute qk ----
        k = tl.load(k_ptrs, block_n_offs[:, None] < N_CTX, 0.)
        if USE_FP8:
            k = k.to(tl.float8e5, bitcast=True)
            k = k.to(tl.float16)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale
        # The score mask uses the same global key coordinate as the K/V loads.
        qk = tl.where(block_n_offs[None, :] < N_CTX, qk, float("-inf"))
        if IS_CAUSAL:  # <- ADDED
            qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + start_position), qk, float("-inf"))

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
        p = p.to(Q.dtype.element_ty)
        v = tl.load(v_ptrs, block_n_offs[:, None] < N_CTX, 0.0)
        if USE_FP8:
            v = v.to(tl.float8e5, bitcast=True)
            v = v.to(tl.float16)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_o = batch_id * stride_oz + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, offs_m[:, None] < H)
  }

def carryFirstHalfKernel
    (Q K V Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel :=
  let USE_FP8 := Bool.false
  with_triton_call halfSignature call_args [stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_vz := stride_vz, stride_vh := stride_vh, stride_vk := stride_vk, stride_vn := stride_vn, stride_oz := stride_oz, stride_oh := stride_oh, stride_om := stride_om, stride_on := stride_on, N_HEAD := N_HEAD, H := H, N_CTX := N_CTX, start_position := start_position] in triton {
    start_m = tl.program_id(0)

    head_idx = tl.program_id(1)
    batch_id = head_idx // N_HEAD
    off_hz = head_idx % N_HEAD

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = batch_id * stride_qz + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk  # <- stride fixed
    off_k = batch_id * stride_kz + off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk  # <- stride fixed
    off_v = batch_id * stride_vz + off_hz * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn  # <- stride fixed
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs, offs_m[:, None] < H, other=0.0)
    # loop over k, v and update accumulator
    block_n_end = N_CTX  # <- ADDED (including the IF)
    if IS_CAUSAL:
        # in causal mode, we expect that BLOCK_M_SIZE == BLOCK_N_SIZE
        # autotune will prune shapes not matching this rule
        block_n_end = (start_m + 1) * BLOCK_N + start_position
    for start_n in range(0, block_n_end, BLOCK_N):
        block_n_offs = start_n + offs_n  # <- ADDED
        # -- compute qk ----
        k = tl.load(k_ptrs, block_n_offs[:, None] < N_CTX, 0.)
        if USE_FP8:
            k = k.to(tl.float8e5, bitcast=True)
            k = k.to(tl.float16)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale
        # The score mask uses the same global key coordinate as the K/V loads.
        qk = tl.where(block_n_offs[None, :] < N_CTX, qk, float("-inf"))
        if IS_CAUSAL:  # <- ADDED
            qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + start_position), qk, float("-inf"))

        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        # correct old l
        l_prev *= tl.exp(m_prev - m_curr)
        # attention weights
        p = tl.exp(qk - m_curr[:, None])
        l_curr = l_prev + tl.sum(p, 1)
        # rescale operands of matmuls
        l_rcp = 1. / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(Q.dtype.element_ty)
        v = tl.load(v_ptrs, block_n_offs[:, None] < N_CTX, 0.0)
        if USE_FP8:
            v = v.to(tl.float8e5, bitcast=True)
            v = v.to(tl.float16)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_o = batch_id * stride_oz + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, offs_m[:, None] < H)
  }

def bytesSignature : CallSignature where
  pointers := [("Q",.fp16),("K",.i8),("V",.i8),("Out",.fp16)]
  scalars := [("sm_scale",.fp32),("stride_qz",.integer),("stride_qh",.integer),("stride_qm",.integer),("stride_qk",.integer),("stride_kz",.integer),("stride_kh",.integer),("stride_kn",.integer),("stride_kk",.integer),("stride_vz",.integer),("stride_vh",.integer),("stride_vk",.integer),("stride_vn",.integer),("stride_oz",.integer),("stride_oh",.integer),("stride_om",.integer),("stride_on",.integer),("N_HEAD",.integer),("H",.integer),("N_CTX",.integer),("start_position",.integer)]
  constexpr := ["IS_CAUSAL","BLOCK_M","BLOCK_N","BLOCK_DMODEL","USE_FP8"]
  readPorts := ["Q","K","V"]
  writePorts := ["Out"]

def bytesKernel
    (Q K V Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel :=
  let USE_FP8 := Bool.true
  with_triton_call bytesSignature call_args [stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_vz := stride_vz, stride_vh := stride_vh, stride_vk := stride_vk, stride_vn := stride_vn, stride_oz := stride_oz, stride_oh := stride_oh, stride_om := stride_om, stride_on := stride_on, N_HEAD := N_HEAD, H := H, N_CTX := N_CTX, start_position := start_position] in triton {
    start_m = tl.program_id(0)

    head_idx = tl.program_id(1)
    batch_id = head_idx // N_HEAD
    off_hz = head_idx % N_HEAD

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = batch_id * stride_qz + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk  # <- stride fixed
    off_k = batch_id * stride_kz + off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk  # <- stride fixed
    off_v = batch_id * stride_vz + off_hz * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn  # <- stride fixed
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs, offs_m[:, None] < H, other=0.0)
    # loop over k, v and update accumulator
    block_n_end = N_CTX  # <- ADDED (including the IF)
    if IS_CAUSAL:
        # in causal mode, we expect that BLOCK_M_SIZE == BLOCK_N_SIZE
        # autotune will prune shapes not matching this rule
        block_n_end = (start_m + 1) * BLOCK_N + start_position
    for start_n in range(0, block_n_end, BLOCK_N):
        block_n_offs = start_n + offs_n  # <- ADDED
        # -- compute qk ----
        k = tl.load(k_ptrs, block_n_offs[:, None] < N_CTX, 0.)
        if USE_FP8:
            k = k.to(tl.float8e5, bitcast=True)
            k = k.to(tl.float16)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale
        # The score mask uses the same global key coordinate as the K/V loads.
        qk = tl.where(block_n_offs[None, :] < N_CTX, qk, float("-inf"))
        if IS_CAUSAL:  # <- ADDED
            qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + start_position), qk, float("-inf"))

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
        p = p.to(Q.dtype.element_ty)
        v = tl.load(v_ptrs, block_n_offs[:, None] < N_CTX, 0.0)
        if USE_FP8:
            v = v.to(tl.float8e5, bitcast=True)
            v = v.to(tl.float16)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_o = batch_id * stride_oz + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, offs_m[:, None] < H)
  }

def carryFirstBytesKernel
    (Q K V Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel :=
  let USE_FP8 := Bool.true
  with_triton_call bytesSignature call_args [stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_vz := stride_vz, stride_vh := stride_vh, stride_vk := stride_vk, stride_vn := stride_vn, stride_oz := stride_oz, stride_oh := stride_oh, stride_om := stride_om, stride_on := stride_on, N_HEAD := N_HEAD, H := H, N_CTX := N_CTX, start_position := start_position] in triton {
    start_m = tl.program_id(0)

    head_idx = tl.program_id(1)
    batch_id = head_idx // N_HEAD
    off_hz = head_idx % N_HEAD

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = batch_id * stride_qz + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk  # <- stride fixed
    off_k = batch_id * stride_kz + off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk  # <- stride fixed
    off_v = batch_id * stride_vz + off_hz * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn  # <- stride fixed
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs, offs_m[:, None] < H, other=0.0)
    # loop over k, v and update accumulator
    block_n_end = N_CTX  # <- ADDED (including the IF)
    if IS_CAUSAL:
        # in causal mode, we expect that BLOCK_M_SIZE == BLOCK_N_SIZE
        # autotune will prune shapes not matching this rule
        block_n_end = (start_m + 1) * BLOCK_N + start_position
    for start_n in range(0, block_n_end, BLOCK_N):
        block_n_offs = start_n + offs_n  # <- ADDED
        # -- compute qk ----
        k = tl.load(k_ptrs, block_n_offs[:, None] < N_CTX, 0.)
        if USE_FP8:
            k = k.to(tl.float8e5, bitcast=True)
            k = k.to(tl.float16)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale
        # The score mask uses the same global key coordinate as the K/V loads.
        qk = tl.where(block_n_offs[None, :] < N_CTX, qk, float("-inf"))
        if IS_CAUSAL:  # <- ADDED
            qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + start_position), qk, float("-inf"))

        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        # correct old l
        l_prev *= tl.exp(m_prev - m_curr)
        # attention weights
        p = tl.exp(qk - m_curr[:, None])
        l_curr = l_prev + tl.sum(p, 1)
        # rescale operands of matmuls
        l_rcp = 1. / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(Q.dtype.element_ty)
        v = tl.load(v_ptrs, block_n_offs[:, None] < N_CTX, 0.0)
        if USE_FP8:
            v = v.to(tl.float8e5, bitcast=True)
            v = v.to(tl.float16)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_o = batch_id * stride_oz + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, offs_m[:, None] < H)
  }
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked
