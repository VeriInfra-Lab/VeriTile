import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Historical generic original/scaleFirst source, retained for compatibility.
The selected FP32 original and address-CSE candidate have complete ordinary
quotations in Kernels.lean. -/
namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile
open Lean VeriTile.Triton
syntax "minDotSimpleGLASource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(minDotSimpleGLASource%[$dotTy:tritonDType, $vTy:tritonDType, $gTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scaleFirst do
      Macro.throwError "expected original or scaleFirst"
    let result : TSyntax `tritonStmt ←
      if version.getId == `scaleFirst then
        `(tritonStmt| b_o = $((scale : ℝ)) * (b_o + tl.dot((b_s).to($vTy:tritonDType), b_v, allow_tf32=False)))
      else
        `(tritonStmt| b_o = (b_o + tl.dot((b_s).to($vTy:tritonDType), b_v, allow_tf32=False)) * $((scale : ℝ)))
    `(fun (out_dtype : FloatDType) (q k v h g o : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV : Nat) =>
      ComputeKernel.mk [q,k,v,h,g] [o] (triton {
        tl.region q = $dotTy, k = $dotTy, h = $dotTy, v = $vTy, g = $gTy, o = OUT_DTYPE
        i_v = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        o_i = tl.arange(0, $(BT))
        m_s = o_i[:, None] >= o_i[None, :]
        b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
        b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
        rows = i_t * $(BT) + tl.arange(0, $(BT))
        cols = i_v * $(BV) + tl.arange(0, $(BV))
        for i_k in range(tl.cdiv($(K), $(BK))) {
          keys = i_k * $(BK) + tl.arange(0, $(BK))
          p_q = q + i_bh * $(s_k_h) + rows[:, None] * $(s_k_t) + keys[None, :]
          p_k = k + i_bh * $(s_k_h) + keys[:, None] + rows[None, :] * $(s_k_t)
          p_h = h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + keys[:, None] * $(s_h_t) + cols[None, :]
          b_q = tl.load(p_q, mask=(rows[:, None] < $(T)) & (keys[None, :] < $(K)), other=0)
          b_k = tl.load(p_k, mask=(keys[:, None] < $(K)) & (rows[None, :] < $(T)), other=0)
          b_h = tl.load(p_h, mask=(keys[:, None] < $(K)) & (cols[None, :] < $(V)), other=0)
          b_o += tl.dot(b_q, b_h, allow_tf32=False)
          b_s += tl.dot(b_q, b_k, allow_tf32=False)
        }
        p_g = g + i_bh * $(T) + rows
        b_g = tl.load(p_g, mask=rows < $(T), other=0)
        b_o = b_o * tl.exp(b_g)[:, None]
        b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
        b_s = tl.where(m_s, b_s, 0)
        p_v = v + i_bh * $(s_v_h) + rows[:, None] * $(s_v_t) + cols[None, :]
        b_v = tl.load(p_v, mask=(rows[:, None] < $(T)) & (cols[None, :] < $(V)), other=0)
        $result
        p_o = o + i_bh * $(s_v_h) + rows[:, None] * $(s_v_t) + cols[None, :]
        tl.store(p_o, (b_o).to(OUT_DTYPE), mask=(rows[:, None] < $(T)) & (cols[None, :] < $(V)))
      }).surfaceBody)
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile
