import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Triton.Float.Control
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.FlashAttn.ActualSource
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

structure Parameters where
  (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim : Nat)
  (stride_k_bs stride_k_head stride_k_seqlen stride_k_dim : Nat)
  (stride_v_bs stride_v_head stride_v_seqlen stride_v_dim : Nat)
  (stride_o_bs stride_o_head stride_o_seqlen stride_o_dim : Nat)
  (BS HEAD SEQLEN DIM : Nat)
  IS_CAUSAL : Bool

/-! Complete repaired FP32 attention source. The incoming scalar is metadata,
not an executed placeholder. The retained candidate commutes the exact natural
head/stride products; every floating expression and all four strides are retained. -/
def kernel (p : Parameters) : ComputeKernel :=
  let Q : RegionName := "Q"
  let K : RegionName := "K"
  let V : RegionName := "V"
  let L : RegionName := "L"
  let O : RegionName := "O"
  let IS_CAUSAL := p.IS_CAUSAL
  triton {
  tl.ports inputs=["Q","K","V"], outputs=["L","O"]
  tl.region Q = tl.float32, K = tl.float32, V = tl.float32, L = tl.float32, O = tl.float32
  tl.param sm_scale : tl.float32
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)
  off_bs = off_bs_head // $((p.HEAD : Nat))
  off_head = off_bs_head % $((p.HEAD : Nat))
  q_base_offset = off_bs * $((p.stride_q_bs : Nat)) + off_head * $((p.stride_q_head : Nat))
  k_base_offset = off_bs * $((p.stride_k_bs : Nat)) + off_head * $((p.stride_k_head : Nat))
  v_base_offset = off_bs * $((p.stride_v_bs : Nat)) + off_head * $((p.stride_v_head : Nat))
  o_base_offset = off_bs * $((p.stride_o_bs : Nat)) + off_head * $((p.stride_o_head : Nat))
  Q_block_ptr = tl.make_block_ptr(base=Q + q_base_offset,
    shape=($((p.SEQLEN : Nat)), $((p.DIM : Nat))), strides=($((p.stride_q_seqlen : Nat)), $((p.stride_q_dim : Nat))),
    offsets=(start_m * $(128), 0), block_shape=(128, $((p.DIM : Nat))), order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_base_offset,
    shape=($((p.DIM : Nat)), $((p.SEQLEN : Nat))), strides=($((p.stride_k_dim : Nat)), $((p.stride_k_seqlen : Nat))),
    offsets=(0, 0), block_shape=($((p.DIM : Nat)), 64), order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + v_base_offset,
    shape=($((p.SEQLEN : Nat)), $((p.DIM : Nat))), strides=($((p.stride_v_seqlen : Nat)), $((p.stride_v_dim : Nat))),
    offsets=(0, 0), block_shape=(64, $((p.DIM : Nat))), order=(1, 0))
  off_m = start_m * $(128) + tl.arange(0, 128)
  off_n = tl.arange(0, 64)
  max = tl.zeros([128], dtype=tl.float32) - float("inf")
  denom = tl.zeros([128], dtype=tl.float32)
  out_buffer = tl.zeros([128, $((p.DIM : Nat))], dtype=tl.float32)
  qk_scale = sm_scale * 1.44269504
  q = tl.load(Q_block_ptr, boundary_check=([0] : List Nat), padding_option="zero")
  q = (q * qk_scale).to(tl.float16)
  lo = 0
  hi = (tl.minimum((start_m + $(1)) * $(128), $((p.SEQLEN : Nat))) if IS_CAUSAL else $((p.SEQLEN : Nat)))
  for start_n in range(lo, hi, 64) {
    k = tl.load(K_block_ptr, boundary_check=([1] : List Nat), padding_option="zero")
    v = tl.load(V_block_ptr, boundary_check=([0] : List Nat), padding_option="zero")
    qk = tl.zeros([128, 64], dtype=tl.float32)
    if IS_CAUSAL {
      qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
    }
    qk = tl.where(start_n + off_n[None, :] < $((p.SEQLEN : Nat)), qk, float("-inf"))
    qk += tl.dot(q.to(tl.float32), k)
    max_new = tl.maximum(max, tl.max(qk, 1))
    alpha = tl.math.exp2(max - max_new)
    nume = tl.math.exp2(qk - max_new[:, None])
    out_scale = denom * 0 + alpha
    out_buffer *= out_scale[:, None]
    out_buffer += tl.dot(nume.to(tl.float16).to(tl.float32), v)
    denom = denom * alpha + tl.sum(nume, 1)
    max = max_new
    K_block_ptr = tl.advance(K_block_ptr, (0, 64))
    V_block_ptr = tl.advance(V_block_ptr, (64, 0))
  }
  out_buffer = out_buffer / denom[:, None]
  l_ptr = L + off_bs_head * $((p.SEQLEN : Nat)) + off_m
  tl.store(l_ptr, max + tl.math.log2(denom), mask=off_m < $((p.SEQLEN : Nat)))
  O_block_ptr = tl.make_block_ptr(base=O + o_base_offset,
    shape=($((p.SEQLEN : Nat)), $((p.DIM : Nat))), strides=($((p.stride_o_seqlen : Nat)), $((p.stride_o_dim : Nat))),
    offsets=(start_m * $(128), 0), block_shape=(128, $((p.DIM : Nat))), order=(1, 0))
  tl.store(O_block_ptr, out_buffer.to(tl.float16).to(tl.float32), boundary_check=([0] : List Nat))
}

/-- Existing integer head-offset normalization, lifted to the repaired independent strides. -/
def indexCommuted (p : Parameters) : ComputeKernel :=
  let Q : RegionName := "Q"
  let K : RegionName := "K"
  let V : RegionName := "V"
  let L : RegionName := "L"
  let O : RegionName := "O"
  let IS_CAUSAL := p.IS_CAUSAL
  triton {
  tl.ports inputs=["Q","K","V"], outputs=["L","O"]
  tl.region Q = tl.float32, K = tl.float32, V = tl.float32, L = tl.float32, O = tl.float32
  tl.param sm_scale : tl.float32
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)
  off_bs = off_bs_head // $((p.HEAD : Nat))
  off_head = off_bs_head % $((p.HEAD : Nat))
  q_base_offset = off_bs * $((p.stride_q_bs : Nat)) + $((p.stride_q_head : Nat)) * off_head
  k_base_offset = off_bs * $((p.stride_k_bs : Nat)) + $((p.stride_k_head : Nat)) * off_head
  v_base_offset = off_bs * $((p.stride_v_bs : Nat)) + $((p.stride_v_head : Nat)) * off_head
  o_base_offset = off_bs * $((p.stride_o_bs : Nat)) + $((p.stride_o_head : Nat)) * off_head
  Q_block_ptr = tl.make_block_ptr(base=Q + q_base_offset,
    shape=($((p.SEQLEN : Nat)), $((p.DIM : Nat))), strides=($((p.stride_q_seqlen : Nat)), $((p.stride_q_dim : Nat))),
    offsets=(start_m * $(128), 0), block_shape=(128, $((p.DIM : Nat))), order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_base_offset,
    shape=($((p.DIM : Nat)), $((p.SEQLEN : Nat))), strides=($((p.stride_k_dim : Nat)), $((p.stride_k_seqlen : Nat))),
    offsets=(0, 0), block_shape=($((p.DIM : Nat)), 64), order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + v_base_offset,
    shape=($((p.SEQLEN : Nat)), $((p.DIM : Nat))), strides=($((p.stride_v_seqlen : Nat)), $((p.stride_v_dim : Nat))),
    offsets=(0, 0), block_shape=(64, $((p.DIM : Nat))), order=(1, 0))
  off_m = start_m * $(128) + tl.arange(0, 128)
  off_n = tl.arange(0, 64)
  max = tl.zeros([128], dtype=tl.float32) - float("inf")
  denom = tl.zeros([128], dtype=tl.float32)
  out_buffer = tl.zeros([128, $((p.DIM : Nat))], dtype=tl.float32)
  qk_scale = sm_scale * 1.44269504
  q = tl.load(Q_block_ptr, boundary_check=([0] : List Nat), padding_option="zero")
  q = (q * qk_scale).to(tl.float16)
  lo = 0
  hi = (tl.minimum((start_m + $(1)) * $(128), $((p.SEQLEN : Nat))) if IS_CAUSAL else $((p.SEQLEN : Nat)))
  for start_n in range(lo, hi, 64) {
    k = tl.load(K_block_ptr, boundary_check=([1] : List Nat), padding_option="zero")
    v = tl.load(V_block_ptr, boundary_check=([0] : List Nat), padding_option="zero")
    qk = tl.zeros([128, 64], dtype=tl.float32)
    if IS_CAUSAL {
      qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
    }
    qk = tl.where(start_n + off_n[None, :] < $((p.SEQLEN : Nat)), qk, float("-inf"))
    qk += tl.dot(q.to(tl.float32), k)
    max_new = tl.maximum(max, tl.max(qk, 1))
    alpha = tl.math.exp2(max - max_new)
    nume = tl.math.exp2(qk - max_new[:, None])
    out_scale = denom * 0 + alpha
    out_buffer *= out_scale[:, None]
    out_buffer += tl.dot(nume.to(tl.float16).to(tl.float32), v)
    denom = denom * alpha + tl.sum(nume, 1)
    max = max_new
    K_block_ptr = tl.advance(K_block_ptr, (0, 64))
    V_block_ptr = tl.advance(V_block_ptr, (64, 0))
  }
  out_buffer = out_buffer / denom[:, None]
  l_ptr = L + off_bs_head * $((p.SEQLEN : Nat)) + off_m
  tl.store(l_ptr, max + tl.math.log2(denom), mask=off_m < $((p.SEQLEN : Nat)))
  O_block_ptr = tl.make_block_ptr(base=O + o_base_offset,
    shape=($((p.SEQLEN : Nat)), $((p.DIM : Nat))), strides=($((p.stride_o_seqlen : Nat)), $((p.stride_o_dim : Nat))),
    offsets=(start_m * $(128), 0), block_shape=(128, $((p.DIM : Nat))), order=(1, 0))
  tl.store(O_block_ptr, out_buffer.to(tl.float16).to(tl.float32), boundary_check=([0] : List Nat))
}

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
