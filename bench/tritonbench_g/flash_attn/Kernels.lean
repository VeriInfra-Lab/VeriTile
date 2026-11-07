import VeriTile.Triton.DSL.PythonOptionalConditionals
import VeriTile.Triton.Source
import VeriTile.Triton.Float.Control
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.FlashAttn.ActualSource
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedVariables false

structure Parameters where
  (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim : Nat)
  (stride_k_bs stride_k_head stride_k_seqlen stride_k_dim : Nat)
  (stride_v_bs stride_v_head stride_v_seqlen stride_v_dim : Nat)
  (stride_o_bs stride_o_head stride_o_seqlen stride_o_dim : Nat)
  (BS HEAD SEQLEN DIM : Nat)
  IS_CAUSAL : Bool

def callSignature : CallSignature where
  pointers := [("Q",.fp32),("K",.fp32),("V",.fp32),("L",.fp32),("O",.fp32)]
  scalars := [("stride_q_bs",.integer),("stride_q_head",.integer),("stride_q_seqlen",.integer),("stride_q_dim",.integer),("stride_k_bs",.integer),("stride_k_head",.integer),("stride_k_seqlen",.integer),("stride_k_dim",.integer),("stride_v_bs",.integer),("stride_v_head",.integer),("stride_v_seqlen",.integer),("stride_v_dim",.integer),("stride_o_bs",.integer),("stride_o_head",.integer),("stride_o_seqlen",.integer),("stride_o_dim",.integer),("BS",.integer),("HEAD",.integer),("SEQLEN",.integer),("sm_scale",.fp32)]
  constexpr := ["BLOCK_M","DIM","BLOCK_N","IS_CAUSAL"]
  readPorts := ["Q","K","V"]
  writePorts := ["L","O"]

def bodyKernel (Q K V L O : RegionName) (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim stride_k_bs stride_k_head stride_k_seqlen stride_k_dim stride_v_bs stride_v_head stride_v_seqlen stride_v_dim stride_o_bs stride_o_head stride_o_seqlen stride_o_dim BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (IS_CAUSAL : Bool) : ComputeKernel :=
  with_triton_call callSignature call_args [stride_q_bs := stride_q_bs, stride_q_head := stride_q_head, stride_q_seqlen := stride_q_seqlen, stride_q_dim := stride_q_dim, stride_k_bs := stride_k_bs, stride_k_head := stride_k_head, stride_k_seqlen := stride_k_seqlen, stride_k_dim := stride_k_dim, stride_v_bs := stride_v_bs, stride_v_head := stride_v_head, stride_v_seqlen := stride_v_seqlen, stride_v_dim := stride_v_dim, stride_o_bs := stride_o_bs, stride_o_head := stride_o_head, stride_o_seqlen := stride_o_seqlen, stride_o_dim := stride_o_dim, BS := BS, HEAD := HEAD, SEQLEN := SEQLEN] in triton {
    start_m = tl.program_id(0)
    off_bs_head = tl.program_id(1)

    off_bs = off_bs_head // HEAD
    off_head = off_bs_head % HEAD
    q_base_offset = off_bs * stride_q_bs + off_head * stride_q_head
    k_base_offset = off_bs * stride_k_bs + off_head * stride_k_head
    v_base_offset = off_bs * stride_v_bs + off_head * stride_v_head
    o_base_offset = off_bs * stride_o_bs + off_head * stride_o_head
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_q_seqlen, stride_q_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_base_offset,
        shape=(DIM, SEQLEN),
        strides=(stride_k_dim, stride_k_seqlen),
        offsets=(0, 0),
        block_shape=(DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_v_seqlen, stride_v_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_N, DIM),
        order=(1, 0),
    )
    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    max = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    out_buffer = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    q = (q * qk_scale).to(tl.float16)
    lo = 0
    hi = tl.minimum((start_m + 1) * BLOCK_M, SEQLEN) if IS_CAUSAL else SEQLEN
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
        qk = tl.where(start_n + off_n[None, :] < SEQLEN, qk, float("-inf"))
        qk += tl.dot(q.to(tl.float32), k)

        max_new = tl.maximum(max, tl.max(qk, 1))
        alpha = tl.math.exp2(max - max_new)
        nume = tl.math.exp2(qk - max_new[:, None])
        out_scale = denom * 0 + alpha
        out_buffer *= out_scale[:, None]
        out_buffer += tl.dot(nume.to(tl.float16).to(tl.float32), v)
        denom = denom * alpha + tl.sum(nume, 1)
        max = max_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    out_buffer = out_buffer / denom[:, None]
    l_ptr = L + off_bs_head * SEQLEN + off_m
    tl.store(l_ptr, max + tl.math.log2(denom), mask=off_m < SEQLEN)
    O_block_ptr = tl.make_block_ptr(
        base=O + o_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_o_seqlen, stride_o_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    tl.store(O_block_ptr, out_buffer.to(tl.float16).to(tl.float32), boundary_check=(0,))
  }

def kernel (p : Parameters) : ComputeKernel :=
  bodyKernel "Q" "K" "V" "L" "O"
    p.stride_q_bs p.stride_q_head p.stride_q_seqlen p.stride_q_dim
    p.stride_k_bs p.stride_k_head p.stride_k_seqlen p.stride_k_dim
    p.stride_v_bs p.stride_v_head p.stride_v_seqlen p.stride_v_dim
    p.stride_o_bs p.stride_o_head p.stride_o_seqlen p.stride_o_dim
    p.BS p.HEAD p.SEQLEN
    128 p.DIM 64 p.IS_CAUSAL

def indexCommutedKernel (Q K V L O : RegionName) (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim stride_k_bs stride_k_head stride_k_seqlen stride_k_dim stride_v_bs stride_v_head stride_v_seqlen stride_v_dim stride_o_bs stride_o_head stride_o_seqlen stride_o_dim BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (IS_CAUSAL : Bool) : ComputeKernel :=
  with_triton_call callSignature call_args [stride_q_bs := stride_q_bs, stride_q_head := stride_q_head, stride_q_seqlen := stride_q_seqlen, stride_q_dim := stride_q_dim, stride_k_bs := stride_k_bs, stride_k_head := stride_k_head, stride_k_seqlen := stride_k_seqlen, stride_k_dim := stride_k_dim, stride_v_bs := stride_v_bs, stride_v_head := stride_v_head, stride_v_seqlen := stride_v_seqlen, stride_v_dim := stride_v_dim, stride_o_bs := stride_o_bs, stride_o_head := stride_o_head, stride_o_seqlen := stride_o_seqlen, stride_o_dim := stride_o_dim, BS := BS, HEAD := HEAD, SEQLEN := SEQLEN] in triton {
    start_m = tl.program_id(0)
    off_bs_head = tl.program_id(1)

    off_bs = off_bs_head // HEAD
    off_head = off_bs_head % HEAD
    q_base_offset = off_bs * stride_q_bs + stride_q_head * off_head
    k_base_offset = off_bs * stride_k_bs + stride_k_head * off_head
    v_base_offset = off_bs * stride_v_bs + stride_v_head * off_head
    o_base_offset = off_bs * stride_o_bs + stride_o_head * off_head
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_q_seqlen, stride_q_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_base_offset,
        shape=(DIM, SEQLEN),
        strides=(stride_k_dim, stride_k_seqlen),
        offsets=(0, 0),
        block_shape=(DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_v_seqlen, stride_v_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_N, DIM),
        order=(1, 0),
    )
    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    max = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    out_buffer = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    q = (q * qk_scale).to(tl.float16)
    lo = 0
    hi = tl.minimum((start_m + 1) * BLOCK_M, SEQLEN) if IS_CAUSAL else SEQLEN
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
        qk = tl.where(start_n + off_n[None, :] < SEQLEN, qk, float("-inf"))
        qk += tl.dot(q.to(tl.float32), k)

        max_new = tl.maximum(max, tl.max(qk, 1))
        alpha = tl.math.exp2(max - max_new)
        nume = tl.math.exp2(qk - max_new[:, None])
        out_scale = denom * 0 + alpha
        out_buffer *= out_scale[:, None]
        out_buffer += tl.dot(nume.to(tl.float16).to(tl.float32), v)
        denom = denom * alpha + tl.sum(nume, 1)
        max = max_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    out_buffer = out_buffer / denom[:, None]
    l_ptr = L + off_bs_head * SEQLEN + off_m
    tl.store(l_ptr, max + tl.math.log2(denom), mask=off_m < SEQLEN)
    O_block_ptr = tl.make_block_ptr(
        base=O + o_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_o_seqlen, stride_o_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    tl.store(O_block_ptr, out_buffer.to(tl.float16).to(tl.float32), boundary_check=(0,))
  }

def indexCommuted (p : Parameters) : ComputeKernel :=
  indexCommutedKernel "Q" "K" "V" "L" "O"
    p.stride_q_bs p.stride_q_head p.stride_q_seqlen p.stride_q_dim
    p.stride_k_bs p.stride_k_head p.stride_k_seqlen p.stride_k_dim
    p.stride_v_bs p.stride_v_head p.stride_v_seqlen p.stride_v_dim
    p.stride_o_bs p.stride_o_head p.stride_o_seqlen p.stride_o_dim
    p.BS p.HEAD p.SEQLEN
    128 p.DIM 64 p.IS_CAUSAL

/-- Compatibility name for existing source decomposition proofs. -/
abbrev declared := kernel

end
end VeriTile.Bench.Repairs.FlashAttn.ActualSource

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked
abbrev kernel := FlashAttn.ActualSource.kernel

/-- The selected scalar-ABI repair retains the complete shared attention source. -/
theorem shared_source (p : FlashAttn.ActualSource.Parameters) :
    kernel p = FlashAttn.ActualSource.kernel p := rfl
#axiomsClean shared_source
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked
