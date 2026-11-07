import VeriTile.Triton.DSL.CastReverseRange
import bench.tritonbench_g.decay_cumsum.Proofs.Sources.Parameters
namespace VeriTile.Bench.Repairs.DecayCumsum
open VeriTile.Triton VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedVariables false

def forwardSignature (gate : Storage) : CallSignature where
  pointers := [("g",match gate with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("g_o",match gate with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("B",.integer),("H",.integer),("T",.integer),("scale",.fp32)]
  constexpr := ["BT","BK","DK"]
  sourceConstants := [("inv_ln2","1.44269504")]
  readPorts := ["g"]
  writePorts := ["g_o"]

def forwardKernel (gate : Storage) (g g_o : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  with_triton_call (forwardSignature gate) call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, B := B, H := H, T := T, scale := scale] in triton {
    i_k = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_go = g_o + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    cum_decay = tl.zeros([BK], dtype=tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)) < DK

    for i in range(tl.cast(BT, tl.int64)):
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        cum_decay += _g * inv_ln2
        tl.store(p_go, cum_decay.to(p_go.dtype.element_ty), mask=mask)
        p_g += DK
        p_go += DK
}

def forwardCandidateKernel (gate : Storage) (g g_o : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  with_triton_call (forwardSignature gate) call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, B := B, H := H, T := T, scale := scale] in triton {
    i_k = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_go = g_o + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    cum_decay = tl.zeros([BK], dtype=tl.float32)
    mask = (BK * i_k + tl.arange(0, BK)) < DK

    for i in range(tl.cast(BT, tl.int64)):
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        cum_decay += _g * inv_ln2
        tl.store(p_go, cum_decay.to(p_go.dtype.element_ty), mask=mask)
        p_g += DK
        p_go += DK
}

def prepareSignature (source gate : Storage) : CallSignature where
  pointers := [("q",match source with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("k",match source with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("g",match gate with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("qg",match source with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("kg",match source with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("B",.integer),("H",.integer),("T",.integer),("scale",.fp32)]
  constexpr := ["BT","BK","DK"]
  readPorts := ["q", "k", "g"]
  writePorts := ["qg", "kg"]

def prepareKernel (source gate : Storage) (q k g qg kg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  with_triton_call (prepareSignature source gate) call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, B := B, H := H, T := T, scale := scale] in triton {
    i_k = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    p_q = q + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_qg = qg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_kg = kg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)

    mask = (i_k * BK + tl.arange(0, BK)) < DK

    last_decay = tl.load(g + i_bh * s_qk_h + (i_c * BT + BT - 1) * DK + i_k * BK + tl.arange(0, BK))

    for i in range(tl.cast(BT, tl.int64)):
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        _q *= tl.math.exp2(_g) * scale
        _k *= tl.math.exp2(last_decay - _g)
        tl.store(p_kg, _k.to(p_kg.dtype.element_ty), mask=mask)
        tl.store(p_qg, _q.to(p_qg.dtype.element_ty), mask=mask)
        p_q += DK
        p_g += DK
        p_k += DK
        p_kg += DK
        p_qg += DK
}

def prepareCandidateKernel (source gate : Storage) (q k g qg kg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  with_triton_call (prepareSignature source gate) call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, B := B, H := H, T := T, scale := scale] in triton {
    i_k = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    p_q = q + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_qg = qg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_kg = kg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)

    mask = (BK * i_k + tl.arange(0, BK)) < DK

    last_decay = tl.load(g + i_bh * s_qk_h + (i_c * BT + BT - 1) * DK + i_k * BK + tl.arange(0, BK))

    for i in range(tl.cast(BT, tl.int64)):
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        _q *= tl.math.exp2(_g) * scale
        _k *= tl.math.exp2(last_decay - _g)
        tl.store(p_kg, _k.to(p_kg.dtype.element_ty), mask=mask)
        tl.store(p_qg, _q.to(p_qg.dtype.element_ty), mask=mask)
        p_q += DK
        p_g += DK
        p_k += DK
        p_kg += DK
        p_qg += DK
}

def backwardSignature (source gate : Storage) : CallSignature where
  pointers := [("dq_inner",match source with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("dq_inter",match source with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("dk_inner",match source with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("dk_inter",match source with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("q",match source with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("k",match source with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("g",match gate with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("dg",match gate with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("B",.integer),("H",.integer),("T",.integer),("scale",.fp32)]
  constexpr := ["BT","BK","DK"]
  readPorts := ["dq_inner", "dq_inter", "dk_inner", "dk_inter", "q", "k", "g"]
  writePorts := ["dq_inter", "dk_inter", "dg"]

def backwardKernel (source gate : Storage) (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  with_triton_call (backwardSignature source gate) call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, B := B, H := H, T := T, scale := scale] in triton {
    i_k = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_g = g + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dg = dg + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dq_inner = dq_inner + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dk_inner = dk_inner + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dq_inter = dq_inter + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dk_inter = dk_inter + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    cum_grad_dg = tl.zeros([BK], dtype=tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)) < DK
    last_g = tl.zeros([BK], dtype=tl.float32)
    for j in range(tl.cast(BT - 1, tl.int64), tl.cast(-1, tl.int64), -1):
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        if j == (BT-1):
            last_g = _g
        _dq1 = tl.load(p_dq_inner, mask=mask, other=0)
        _dq2 = tl.load(p_dq_inter, mask=mask, other=0)
        _dq2 *= tl.math.exp2(_g)
        _dq = _dq1 + _dq2
        tl.store(p_dq_inter, _dq, mask=mask)
        _dk1 = tl.load(p_dk_inner, mask=mask, other=0)
        _dk2 = tl.load(p_dk_inter, mask=mask, other=0)
        _dk2 *= tl.math.exp2(last_g - _g)
        _dk = _dk1 + _dk2
        tl.store(p_dk_inter, _dk, mask=mask)
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _dg = _dq * _q - _dk * _k
        cum_grad_dg += _dg
        tl.store(p_dg, cum_grad_dg.to(p_dg.dtype.element_ty), mask=mask)
        p_g -= DK
        p_k -= DK
        p_q -= DK
        p_dq_inner -= DK
        p_dk_inner -= DK
        p_dq_inter -= DK
        p_dk_inter -= DK
        p_dg -= DK
}

def backwardCandidateKernel (source gate : Storage) (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  with_triton_call (backwardSignature source gate) call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, B := B, H := H, T := T, scale := scale] in triton {
    i_k = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_g = g + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dg = dg + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dq_inner = dq_inner + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dk_inner = dk_inner + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dq_inter = dq_inter + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dk_inter = dk_inter + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    cum_grad_dg = tl.zeros([BK], dtype=tl.float32)
    mask = (BK * i_k + tl.arange(0, BK)) < DK
    last_g = tl.zeros([BK], dtype=tl.float32)
    for j in range(tl.cast(BT - 1, tl.int64), tl.cast(-1, tl.int64), -1):
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        if j == (BT-1):
            last_g = _g
        _dq1 = tl.load(p_dq_inner, mask=mask, other=0)
        _dq2 = tl.load(p_dq_inter, mask=mask, other=0)
        _dq2 *= tl.math.exp2(_g)
        _dq = _dq1 + _dq2
        tl.store(p_dq_inter, _dq, mask=mask)
        _dk1 = tl.load(p_dk_inner, mask=mask, other=0)
        _dk2 = tl.load(p_dk_inter, mask=mask, other=0)
        _dk2 *= tl.math.exp2(last_g - _g)
        _dk = _dk1 + _dk2
        tl.store(p_dk_inter, _dk, mask=mask)
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _dg = _dq * _q - _dk * _k
        cum_grad_dg += _dg
        tl.store(p_dg, cum_grad_dg.to(p_dg.dtype.element_ty), mask=mask)
        p_g -= DK
        p_k -= DK
        p_q -= DK
        p_dq_inner -= DK
        p_dk_inner -= DK
        p_dq_inter -= DK
        p_dk_inter -= DK
        p_dg -= DK
}

end
end VeriTile.Bench.Repairs.DecayCumsum
