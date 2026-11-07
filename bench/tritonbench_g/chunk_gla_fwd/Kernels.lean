import VeriTile.Triton.DSL.PastedRangeLoads

/-! Complete repaired Python bodies. Caller signatures supply dtype/argument
metadata; every source cast, mask, early return, loop and store is retained.
The complete existing candidates are checked against their Python source. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedOutput
open VeriTile.Triton
inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

def Storage.dtype : Storage → FloatDType
  | .fp32=>.fp32 | .fp16=>.fp16 | .bf16=>.bf16

def Storage.sourceDType : Storage → SourceDType
  | .fp32=>.fp32 | .fp16=>.fp16 | .bf16=>.bf16
end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedOutput

namespace VeriTile.Bench.Repairs.ChunkGlaFwd
open VeriTile.Triton
abbrev Storage := TypedOutput.Storage
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000

def interSignature (qStorage kStorage gStorage : Storage) (out_dtype : FloatDType) : CallSignature where
  pointers := [("q",(match qStorage with | .fp32=>.fp32 | .fp16=>.fp16 | .bf16=>.bf16)),("k",(match kStorage with | .fp32=>.fp32 | .fp16=>.fp16 | .bf16=>.bf16)),("g",(match gStorage with | .fp32=>.fp32 | .fp16=>.fp16 | .bf16=>.bf16))]
  floatingPointers := [("A",out_dtype)]
  scalars := [("s_k_h",.integer),("s_k_t",.integer),("scale",.fp32)]
  constexpr := ["T","K","BT","BC","BK","NC"]
  readPorts := ["q","k","g"]
  writePorts := ["A"]

def interOriginalKernel (qStorage kStorage gStorage : Storage) (out_dtype : FloatDType) (q k g A : RegionName) (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat) : ComputeKernel :=
  with_triton_call (interSignature qStorage kStorage gStorage out_dtype) call_args [s_k_h := s_k_h, s_k_t := s_k_t, scale := scale] in triton {
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_i, i_j = i_c // NC, i_c % NC
    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = g + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero")
        b_qg = b_q * tl.exp(b_g - b_gn[None, :]) * scale
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_gk = tl.load(p_gk, boundary_check=(0, 1), padding_option="zero")
        b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_qg, b_kg)

    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
  }

def interDotFirstKernel (qStorage kStorage gStorage : Storage) (out_dtype : FloatDType) (q k g A : RegionName) (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat) : ComputeKernel :=
  with_triton_call (interSignature qStorage kStorage gStorage out_dtype) call_args [s_k_h := s_k_h, s_k_t := s_k_t, scale := scale] in triton {
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_i, i_j = i_c // NC, i_c % NC
    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = g + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero")
        b_qg = b_q * tl.exp(b_g - b_gn[None, :]) * scale
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_gk = tl.load(p_gk, boundary_check=(0, 1), padding_option="zero")
        b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
        b_A = tl.dot(b_qg, b_kg) + b_A

    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
  }

def intraSignature (qStorage kStorage gStorage : Storage) : CallSignature where
  pointers := [("q",(match qStorage with | .fp32=>.fp32 | .fp16=>.fp16 | .bf16=>.bf16)),("k",(match kStorage with | .fp32=>.fp32 | .fp16=>.fp16 | .bf16=>.bf16)),("g",(match gStorage with | .fp32=>.fp32 | .fp16=>.fp16 | .bf16=>.bf16)),("A",.fp32)]
  scalars := [("s_k_h",.integer),("s_k_t",.integer),("scale",.fp32)]
  constexpr := ["T","K","BT","BC","BK"]
  readPorts := ["q","k","g"]
  writePorts := ["A"]

def intraOriginalKernel (qStorage kStorage gStorage : Storage) (q k g A : RegionName) (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK : Nat) : ComputeKernel :=
  with_triton_call (intraSignature qStorage kStorage gStorage) call_args [s_k_h := s_k_h, s_k_t := s_k_t, scale := scale] in triton {
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_j = i_i
    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))

    p_k = k + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k
    p_gk = g + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero")
    for j in range(0, min(BC, T-i_t*BT-i_i*BC)):
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, b_A * scale, 0.)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += K
        p_gk += K
  }

def intraScaleFirstKernel (qStorage kStorage gStorage : Storage) (q k g A : RegionName) (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK : Nat) : ComputeKernel :=
  with_triton_call (intraSignature qStorage kStorage gStorage) call_args [s_k_h := s_k_h, s_k_t := s_k_t, scale := scale] in triton {
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_j = i_i
    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))

    p_k = k + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k
    p_gk = g + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero")
    for j in range(0, min(BC, T-i_t*BT-i_i*BC)):
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, scale * b_A, 0.)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += K
        p_gk += K
  }

def splitSignature (qStorage kStorage gStorage : Storage) : CallSignature where
  pointers := [("q",(match qStorage with | .fp32=>.fp32 | .fp16=>.fp16 | .bf16=>.bf16)),("k",(match kStorage with | .fp32=>.fp32 | .fp16=>.fp16 | .bf16=>.bf16)),("g",(match gStorage with | .fp32=>.fp32 | .fp16=>.fp16 | .bf16=>.bf16)),("A",.fp32)]
  scalars := [("s_k_h",.integer),("s_k_t",.integer),("scale",.fp32)]
  constexpr := ["T","K","BT","BC","BK","NC"]
  readPorts := ["q","k","g"]
  writePorts := ["A"]

def splitOriginalKernel (qStorage kStorage gStorage : Storage) (q k g A : RegionName) (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat) : ComputeKernel :=
  with_triton_call (splitSignature qStorage kStorage gStorage) call_args [s_k_h := s_k_h, s_k_t := s_k_t, scale := scale] in triton {
    i_k, i_tc, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_tc // NC, i_tc % NC
    i_j = i_i
    n_bh = tl.num_programs(2)
    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = i_k * BK + tl.arange(0, BK)
    o_A = (i_bh + i_k * n_bh) * T * BC + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BC
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_k = k + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k
    p_gk = g + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero")
    for j in range(0, min(BC, T-i_t*BT-i_i*BC)):
        b_A = tl.zeros([BC], dtype=tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, b_A * scale, 0.)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += K
        p_gk += K
  }

def splitScaleFirstKernel (qStorage kStorage gStorage : Storage) (q k g A : RegionName) (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat) : ComputeKernel :=
  with_triton_call (splitSignature qStorage kStorage gStorage) call_args [s_k_h := s_k_h, s_k_t := s_k_t, scale := scale] in triton {
    i_k, i_tc, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_tc // NC, i_tc % NC
    i_j = i_i
    n_bh = tl.num_programs(2)
    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = i_k * BK + tl.arange(0, BK)
    o_A = (i_bh + i_k * n_bh) * T * BC + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BC
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_k = k + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k
    p_gk = g + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero")
    for j in range(0, min(BC, T-i_t*BT-i_i*BC)):
        b_A = tl.zeros([BC], dtype=tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, scale * b_A, 0.)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += K
        p_gk += K
  }

def mergeSignature (out_dtype : FloatDType) : CallSignature where
  scalars := []
  pointers := [("A",.fp32)]
  floatingPointers := [("A2",out_dtype)]
  constexpr := ["T","BT","BC","NK"]
  readPorts := ["A"]
  writePorts := ["A2"]

def mergeOriginalKernel (out_dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) : ComputeKernel :=
  with_triton_call (mergeSignature out_dtype) call_args [] in triton {
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if i_t * BT + i_c * BC >= T:
        return
    n_bh = tl.num_programs(2)
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(0, NK):
        p_A = tl.make_block_ptr(A + (i_bh + i_k*n_bh) * T * BC, (T, BC), (BC, 1), (i_t * BT + i_c * BC, 0), (BC, BC), (1, 0))
        b_A += tl.load(p_A, boundary_check=(0, 1), padding_option="zero")
    p_A2 = tl.make_block_ptr(A2 + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_c * BC, i_c * BC), (BC, BC), (1, 0))
    tl.store(p_A2, b_A.to(A2.dtype.element_ty), boundary_check=(0, 1))
  }

def mergeLoadFirstKernel (out_dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) : ComputeKernel :=
  with_triton_call (mergeSignature out_dtype) call_args [] in triton {
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if i_t * BT + i_c * BC >= T:
        return
    n_bh = tl.num_programs(2)
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(0, NK):
        p_A = tl.make_block_ptr(A + (i_bh + i_k*n_bh) * T * BC, (T, BC), (BC, 1), (i_t * BT + i_c * BC, 0), (BC, BC), (1, 0))
        b_A = tl.load(p_A, boundary_check=(0, 1), padding_option="zero") + b_A
    p_A2 = tl.make_block_ptr(A2 + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_c * BC, i_c * BC), (BC, BC), (1, 0))
    tl.store(p_A2, b_A.to(A2.dtype.element_ty), boundary_check=(0, 1))
  }

def outputSignature (queryStorage valueStorage gateStorage : Storage) (dtype out_dtype : FloatDType) : CallSignature where
  pointers := [("q",(match queryStorage with | .fp32=>.fp32 | .fp16=>.fp16 | .bf16=>.bf16)),("v",(match valueStorage with | .fp32=>.fp32 | .fp16=>.fp16 | .bf16=>.bf16)),("g",(match gateStorage with | .fp32=>.fp32 | .fp16=>.fp16 | .bf16=>.bf16))]
  floatingPointers := [("h",dtype),("o",out_dtype),("A",dtype)]
  scalars := [("s_k_h",.integer),("s_k_t",.integer),("s_v_h",.integer),("s_v_t",.integer),("s_h_h",.integer),("s_h_t",.integer),("scale",.fp32)]
  constexpr := ["T","K","V","BT","BK","BV"]
  readPorts := ["q","v","g","h","A"]
  writePorts := ["o"]

def outputOriginalKernel (queryStorage valueStorage gateStorage : Storage) (dtype out_dtype : FloatDType) (q v g h o A : RegionName) (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat) : ComputeKernel :=
  with_triton_call (outputSignature queryStorage valueStorage gateStorage dtype out_dtype) call_args [s_k_h := s_k_h, s_k_t := s_k_t, s_v_h := s_v_h, s_v_t := s_v_t, s_h_h := s_h_h, s_h_t := s_h_t, scale := scale] in triton {
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_q = (b_q * scale).to(b_q.dtype)
        b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero")
        b_qg = (b_q * tl.exp(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))

    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_A = tl.load(p_A, boundary_check=(0, 1), padding_option="zero")
    b_A = tl.where(m_s, b_A, 0.).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
  }

def outputSumLeftKernel (queryStorage valueStorage gateStorage : Storage) (dtype out_dtype : FloatDType) (q v g h o A : RegionName) (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat) : ComputeKernel :=
  with_triton_call (outputSignature queryStorage valueStorage gateStorage dtype out_dtype) call_args [s_k_h := s_k_h, s_k_t := s_k_t, s_v_h := s_v_h, s_v_t := s_v_t, s_h_h := s_h_h, s_h_t := s_h_t, scale := scale] in triton {
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_q = (b_q * scale).to(b_q.dtype)
        b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero")
        b_qg = (b_q * tl.exp(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))

    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_A = tl.load(p_A, boundary_check=(0, 1), padding_option="zero")
    b_A = tl.where(m_s, b_A, 0.).to(b_v.dtype)
    b_o = tl.dot(b_A, b_v, allow_tf32=False) + b_o
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
  }

end VeriTile.Bench.Repairs.ChunkGlaFwd

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedInter
open VeriTile.Triton
abbrev Storage := TypedOutput.Storage
def fp32Original := interOriginalKernel .fp32 .fp32 .fp32
def fp32DotFirst := interDotFirstKernel .fp32 .fp32 .fp32
def source (qStorage kStorage gStorage : Storage) (swapped : Bool) :=
  if swapped then interDotFirstKernel qStorage kStorage gStorage
  else interOriginalKernel qStorage kStorage gStorage
def original (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.false
def dotFirst (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.true
end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedInter

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedIntra
open VeriTile.Triton
abbrev Storage := TypedOutput.Storage
def fp32Original := intraOriginalKernel .fp32 .fp32 .fp32
def fp32ScaleFirst := intraScaleFirstKernel .fp32 .fp32 .fp32
def source (qStorage kStorage gStorage : Storage) (swapped : Bool) :=
  if swapped then intraScaleFirstKernel qStorage kStorage gStorage
  else intraOriginalKernel qStorage kStorage gStorage
def original (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.false
def scaleFirst (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.true
end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedIntra

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedSplit
open VeriTile.Triton
abbrev Storage := TypedOutput.Storage
def fp32Original := splitOriginalKernel .fp32 .fp32 .fp32
def fp32ScaleFirst := splitScaleFirstKernel .fp32 .fp32 .fp32
def source (qStorage kStorage gStorage : Storage) (swapped : Bool) :=
  if swapped then splitScaleFirstKernel qStorage kStorage gStorage
  else splitOriginalKernel qStorage kStorage gStorage
def original (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.false
def scaleFirst (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.true
end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedSplit

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedOutput
open VeriTile.Triton
def fp32Original := outputOriginalKernel .fp32 .fp32 .fp32
def fp32SumLeft := outputSumLeftKernel .fp32 .fp32 .fp32
def source (queryStorage valueStorage gateStorage : Storage) (swapped : Bool) :=
  if swapped then outputSumLeftKernel queryStorage valueStorage gateStorage
  else outputOriginalKernel queryStorage valueStorage gateStorage
def original (queryStorage valueStorage gateStorage : Storage) := source queryStorage valueStorage gateStorage Bool.false
def sumLeft (queryStorage valueStorage gateStorage : Storage) := source queryStorage valueStorage gateStorage Bool.true
end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedOutput

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedMerge
def original := mergeOriginalKernel
def loadFirst := mergeLoadFirstKernel
end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedMerge
