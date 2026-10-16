/-
Source-audit target: fused_recurrent_retention_fwd_kernel
Source-audit Lean: retentionForwardSource%
Source-audit args: qTy=tl.float32, kTy=tl.float32, vTy=tl.float32, oTy=tl.float32, seedTy=tl.float32, finalTy=tl.float32, version=original

This inventory selector selects the complete forward fp32 source. The separate
scripts/check_fused_retention_sources.py comparison covers both complete JITs
at every named common-input/independent-output storage pair; one selector must
not be reported as a two-kernel source audit.
-/
import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Complete source macros for both launched retention kernels. Every source
statement, storage cast, mask, flag, pointer update and barrier is retained.
Only the unused Python loop binder `_` is named `_t` in the Lean syntax.
The candidate reverses the operands of the discrete key-mask comparison;
all floating arithmetic and its ordering remain identical.

Each source storage has an independent macro dtype argument. The named public
constructors below instantiate shared input storage and independent output
storage. Other macro profiles remain available without a claim that their
execution proofs have already been replayed. Runtime ABI, native integer
widths and host tensor layout are separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedSource
open Lean VeriTile.Triton

syntax "retentionForwardSource%" "[" tritonDType "," tritonDType "," tritonDType ","
  tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionForwardSource%[$qTy:tritonDType, $kTy:tritonDType, $vTy:tritonDType,
      $oTy:tritonDType, $seedTy:tritonDType, $finalTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `compareRight do
      Macro.throwError "expected original or compareRight"
    let keyMask : TSyntax `tritonStmt ←
      if version.getId == `compareRight then
        `(tritonStmt| mask_bk = $(DK) > (i_k * $(BK) + tl.arange(0, $(BK))))
      else
        `(tritonStmt| mask_bk = (i_k * $(BK) + tl.arange(0, $(BK))) < $(DK))
    `(fun (dtype out_dtype : FloatDType)
        (q k v o initial_state final_state : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
        (BK BV DK DV : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
      ComputeKernel.mk [q, k, v, initial_state] [o, final_state] (triton {
        tl.region q = $qTy, k = $kTy, v = $vTy, o = $oTy,
          initial_state = $seedTy, final_state = $finalTy
        i_v = tl.program_id(0)
        i_k = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = (1 - tl.math.exp2(-5 - i_h * 1.0))
      
        p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
        p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
        p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
        p_o = o + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
      
        $keyMask
        mask_bv = (i_v * $(BV) + tl.arange(0, $(BV))) < $(DV)
        mask_kv = mask_bk[None, :] & mask_bv[:, None]
      
        h = tl.zeros([$(BV), $(BK)], dtype=tl.float32)
      
        if USE_INITIAL_STATE {
          p_init_s = initial_state + i_bh * $(DK) * $(DV) +
            (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(DV) +
            (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
          h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)
        }
      
        for _t in range($(0), $(T), $(1)) {
          _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
          _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
          _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * $(scale)
      
          h = b_b * h + _k[None, :] * _v[:, None]
          _o = h * _q[None, :]
          _o = tl.sum(_o, axis=1)
          tl.store(p_o, (_o).to(p_o.dtype.element_ty), mask=mask_bv)
      
          p_q += $(DK)
          p_k += $(DK)
          p_o += $(DV)
          p_v += $(DV)
        }
      
        if STORE_FINAL_STATE {
          p_final_s = final_state + i_bh * $(DK) * $(DV) +
            (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(DV) +
            (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
          tl.store(p_final_s, (h).to(p_final_s.dtype.element_ty), mask=mask_kv)
        }
      }).surfaceBody)

syntax "retentionBackwardSource%" "[" tritonDType "," tritonDType "," tritonDType ","
  tritonDType "," tritonDType "," tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionBackwardSource%[$qTy:tritonDType, $kTy:tritonDType, $vTy:tritonDType,
      $doTy:tritonDType, $dqTy:tritonDType, $dkTy:tritonDType, $dvTy:tritonDType,
      $seedTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `compareRight do
      Macro.throwError "expected original or compareRight"
    let keyMask : TSyntax `tritonStmt ←
      if version.getId == `compareRight then
        `(tritonStmt| mask_bk = $(DK) > (i_k * $(BK) + tl.arange(0, $(BK))))
      else
        `(tritonStmt| mask_bk = i_k * $(BK) + tl.arange(0, $(BK)) < $(DK))
    `(fun (dtype out_dtype : FloatDType)
        (q k v do_ dq dk dv initial_state : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
        (BK BV DK DV : Nat) (USE_INITIAL_STATE : Bool) =>
      ComputeKernel.mk [q, k, v, do_, initial_state] [dq, dk, dv] (triton {
        tl.region q = $qTy, k = $kTy, v = $vTy, do_ = $doTy,
          dq = $dqTy, dk = $dkTy, dv = $dvTy, initial_state = $seedTy
        i_v = tl.program_id(0)
        i_k = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)
      
        p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
        p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
        p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
        p_do = do_ + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
        p_dq = dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
      
        $keyMask
        mask_bv = i_v * $(BV) + tl.arange(0, $(BV)) < $(DV)
      
        h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
      
        if USE_INITIAL_STATE {
          mask_kv = mask_bk[:, None] & mask_bv[None, :]
          p_init_s = initial_state + i_bh * $(DK) * $(DV) +
            (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(DV) +
            (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
          h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)
        }
      
        for i in range($(0), $(T), $(1)) {
          _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
          _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
          _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
      
          h = b_b * h + _k[:, None] * _v[None, :]
          _d_q = h * _do[None, :]
          d_q = tl.sum(_d_q, axis=1) * $(scale)
          tl.store(p_dq, (d_q).to(p_dq.dtype.element_ty), mask=mask_bk)
      
          p_k += $(DK)
          p_do += $(DV)
          p_v += $(DV)
          p_dq += $(DK)
        }
      
        tl.debug_barrier()
      
        p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - 1) * $(DK)
        p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - 1) * $(DK)
        p_do = do_ + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - 1) * $(DV)
        p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - 1) * $(DV)
        p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - 1) * $(DK)
        p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - 1) * $(DV)
        d_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
      
        for _t in range($(0), $(T), $(1)) {
          _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
          _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * $(scale)
          _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
          _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
          d_h += _q[:, None] * _do[None, :]
          d_k = tl.sum(d_h * _v[None, :], axis=1)
          d_v = tl.sum(d_h * _k[:, None], axis=0)
      
          d_h *= b_b
          tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
          tl.store(p_dv, (d_v).to(p_dv.dtype.element_ty), mask=mask_bv)
      
          p_do -= $(DV)
          p_q -= $(DK)
          p_k -= $(DK)
          p_v -= $(DV)
          p_dk -= $(DK)
          p_dv -= $(DV)
        }
      }).surfaceBody)

def forwardOriginal := retentionForwardSource%[DTYPE, DTYPE, DTYPE,
  OUT_DTYPE, DTYPE, OUT_DTYPE, original]
def forwardCandidate := retentionForwardSource%[DTYPE, DTYPE, DTYPE,
  OUT_DTYPE, DTYPE, OUT_DTYPE, compareRight]
def forward (swapped : Bool) := if swapped then forwardCandidate else forwardOriginal

def backwardOriginal := retentionBackwardSource%[DTYPE, DTYPE, DTYPE, DTYPE,
  OUT_DTYPE, OUT_DTYPE, OUT_DTYPE, DTYPE, original]
def backwardCandidate := retentionBackwardSource%[DTYPE, DTYPE, DTYPE, DTYPE,
  OUT_DTYPE, OUT_DTYPE, OUT_DTYPE, DTYPE, compareRight]
def backward (swapped : Bool) := if swapped then backwardCandidate else backwardOriginal

end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedSource
