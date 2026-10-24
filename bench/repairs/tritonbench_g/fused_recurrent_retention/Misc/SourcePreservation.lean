import bench.repairs.tritonbench_g.fused_recurrent_retention.Kernels
import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Complete repaired FP32 JIT quotations. The first template assignment only
introduces the launch scale register to quotation typing and is removed from
the actual body. Initial-state and final-cotangent strides are retained.
These are source declarations; no execution or expected-output premise occurs. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.LegacySurface
open Lean VeriTile.Triton

syntax "legacyRetentionForwardSource%" "[" tritonDType "," tritonDType "," tritonDType ","
  tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(legacyRetentionForwardSource%[$qTy:tritonDType, $kTy:tritonDType, $vTy:tritonDType,
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
      ComputeKernel.mk [q, k, v, initial_state] [o, final_state] ((triton {
        tl.region q = $qTy, k = $kTy, v = $vTy, o = $oTy,
          initial_state = $seedTy, final_state = $finalTy
        scale = $((0 : ℝ)).to(tl.float32)
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
          _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
      
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
      }).surfaceBody.drop 1))

syntax "legacyRetentionBackwardSource%" "[" tritonDType "," tritonDType "," tritonDType ","
  tritonDType "," tritonDType "," tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(legacyRetentionBackwardSource%[$qTy:tritonDType, $kTy:tritonDType, $vTy:tritonDType,
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
        (q k v do_ dq dk dv initial_state d_final_state : RegionName)
        (s_do_b s_do_h s_do_t s_do_d s_df_b s_df_h s_df_k s_df_v : Nat)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
        (BK BV DK DV : Nat) (USE_INITIAL_STATE USE_FINAL_GRAD : Bool) =>
      ComputeKernel.mk [q, k, v, do_, initial_state, d_final_state] [dq, dk, dv] ((triton {
        tl.region q = $qTy, k = $kTy, v = $vTy, do_ = $doTy,
          dq = $dqTy, dk = $dkTy, dv = $dvTy, initial_state = $seedTy, d_final_state = $seedTy
        scale = $((0 : ℝ)).to(tl.float32)
        i_v = tl.program_id(0)
        i_k = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)
      
        p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
        p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
        p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
        p_do = do_ + (i_bh // $(H)) * $(s_do_b) + i_h * $(s_do_h) + (i_v * $(BV) + tl.arange(0, $(BV))) * $(s_do_d)
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
          d_q = tl.sum(_d_q, axis=1) * scale
          tl.store(p_dq, (d_q).to(p_dq.dtype.element_ty), mask=mask_bk)
      
          p_k += $(DK)
          p_do += $(s_do_t)
          p_v += $(DV)
          p_dq += $(DK)
        }
      
        tl.debug_barrier()
      
        p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - 1) * $(DK)
        p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - 1) * $(DK)
        p_do = do_ + (i_bh // $(H)) * $(s_do_b) + i_h * $(s_do_h) + (i_v * $(BV) + tl.arange(0, $(BV))) * $(s_do_d) + ($(T) - 1) * $(s_do_t)
        p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - 1) * $(DV)
        p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - 1) * $(DK)
        p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - 1) * $(DV)
        d_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_FINAL_GRAD {
          p_df = d_final_state + (i_bh // $(H)) * $(s_df_b) + i_h * $(s_df_h) +
            (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(s_df_k) +
            (i_v * $(BV) + tl.arange(0, $(BV))[None, :]) * $(s_df_v)
          d_h = tl.load(p_df, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)
        }
      
        for _t in range($(0), $(T), $(1)) {
          _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
          _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
          _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
          _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
          d_h += _q[:, None] * _do[None, :]
          d_k = tl.sum(d_h * _v[None, :], axis=1)
          d_v = tl.sum(d_h * _k[:, None], axis=0)
      
          d_h *= b_b
          tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
          tl.store(p_dv, (d_v).to(p_dv.dtype.element_ty), mask=mask_bv)
      
          p_do -= $(s_do_t)
          p_q -= $(DK)
          p_k -= $(DK)
          p_v -= $(DV)
          p_dk -= $(DK)
          p_dv -= $(DV)
        }
      }).surfaceBody.drop 1))

def forward := legacyRetentionForwardSource%[tl.float32, tl.float32, tl.float32,
  tl.float32, tl.float32, tl.float32, original]
def backward := legacyRetentionBackwardSource%[tl.float32, tl.float32, tl.float32, tl.float32,
  tl.float32, tl.float32, tl.float32, tl.float32, original]

def forwardCompareRight := legacyRetentionForwardSource%[tl.float32, tl.float32, tl.float32,
  tl.float32, tl.float32, tl.float32, compareRight]
def backwardCompareRight := legacyRetentionBackwardSource%[tl.float32, tl.float32, tl.float32, tl.float32,
  tl.float32, tl.float32, tl.float32, tl.float32, compareRight]

end VeriTile.Bench.Repairs.FusedRecurrentRetention.LegacySurface

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
/-- Full functional source identity, including ports, scopes and runtime ABI. -/
theorem forward_source_preserved : Surface.forward = LegacySurface.forward := by rfl
theorem backward_source_preserved : Surface.backward = LegacySurface.backward := by rfl
theorem forward_candidate_preserved : Surface.forwardCompareRight = LegacySurface.forwardCompareRight := by rfl
theorem backward_candidate_preserved : Surface.backwardCompareRight = LegacySurface.backwardCompareRight := by rfl
#axiomsClean forward_source_preserved
#axiomsClean backward_source_preserved
#axiomsClean forward_candidate_preserved
#axiomsClean backward_candidate_preserved
end VeriTile.Bench.Repairs.FusedRecurrentRetention
