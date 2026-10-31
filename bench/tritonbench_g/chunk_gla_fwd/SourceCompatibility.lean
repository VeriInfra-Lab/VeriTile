import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Shared syntax for other storage profiles of the current repair.
The selected FP32 source and current candidate are complete ordinary quotations
in Kernels.lean. These profiles use the same repaired padding and layout. -/

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedInter
open Lean VeriTile.Triton
syntax "glaInterSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaInterSource%[$qTy:tritonDType, $kTy:tritonDType, $gTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotFirst do
      Macro.throwError "expected original or dotFirst"
    let add : TSyntax `tritonStmt ←
      if version.getId == `dotFirst then `(tritonStmt| b_A = tl.dot(b_qg, b_kg) + b_A)
      else `(tritonStmt| b_A += tl.dot(b_qg, b_kg))
    `(fun (out_dtype : FloatDType) (q k g A : RegionName)
        (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat) =>
      ComputeKernel.mk [q,k,g] [A] (triton {
        tl.region q = $qTy, k = $kTy, g = $gTy, A = OUT_DTYPE
        i_t = tl.program_id(0)
        i_c = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_i = i_c // $(NC)
        i_j = i_c % $(NC)
        if i_t * $(BT) + i_i * $(BC) >= $(T) { return }
        if i_i <= i_j { return }
        b_A = tl.zeros([$(BC), $(BC)], dtype=tl.float32)
        for i_k in range(tl.cdiv($(K), $(BK))) {
          o_k = i_k * $(BK) + tl.arange(0, $(BK))
          m_k = o_k < $(K)
          p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
            shape=($(T), $(K)), strides=($(s_k_t), $(1)),
            offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)),
            block_shape=($(BC), $(BK)), order=(1, 0))
          p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
            shape=($(T), $(K)), strides=($(s_k_t), $(1)),
            offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)),
            block_shape=($(BC), $(BK)), order=(1, 0))
          p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
            shape=($(K), $(T)), strides=($(1), $(s_k_t)),
            offsets=(i_k * $(BK), i_t * $(BT) + i_j * $(BC)),
            block_shape=($(BK), $(BC)), order=(0, 1))
          p_gk = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
            shape=($(K), $(T)), strides=($(1), $(s_k_t)),
            offsets=(i_k * $(BK), i_t * $(BT) + i_j * $(BC)),
            block_shape=($(BK), $(BC)), order=(0, 1))
          p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * $(s_k_h) +
            (i_t * $(BT) + i_i * $(BC)) * $(K) + o_k, $(BK)), $(BK))
          b_gn = tl.load(p_gn, mask=m_k, other=0)
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_qg = b_q * tl.exp(b_g - b_gn[None, :]) * $((scale : ℝ))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_gk = tl.load(p_gk, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
          $add
        }
        p_A = tl.make_block_ptr(base=A + i_bh * $(T) * $(BT),
          shape=($(T), $(BT)), strides=($(BT), $(1)),
          offsets=(i_t * $(BT) + i_i * $(BC), i_j * $(BC)),
          block_shape=($(BC), $(BC)), order=(1, 0))
        tl.store(p_A, (b_A).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedInter

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedIntra
open Lean VeriTile.Triton
syntax "glaIntraSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaIntraSource%[$qTy:tritonDType, $kTy:tritonDType, $gTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scaleFirst do
      Macro.throwError "expected original or scaleFirst"
    let scaled : TSyntax `tritonStmt ←
      if version.getId == `scaleFirst then `(tritonStmt| b_A = tl.where(o_i >= j, $((scale : ℝ)) * b_A, 0.0))
      else `(tritonStmt| b_A = tl.where(o_i >= j, b_A * $((scale : ℝ)), 0.0))
    `(fun (q k g A : RegionName)
        (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK : Nat) =>
      ComputeKernel.mk [q,k,g] [A] (triton {
        tl.region q = $qTy, k = $kTy, g = $gTy, A = tl.float32
        i_t = tl.program_id(0)
        i_i = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_j = i_i
        if i_t * $(BT) + i_i * $(BC) >= $(T) { return }
        o_i = tl.arange(0, $(BC))
        o_k = tl.arange(0, $(BK))
        o_A = i_bh * $(T) * $(BT) + (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) * $(BT) + i_j * $(BC)
        m_k = o_k < $(K)
        m_A = (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) < $(T)
        p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
          shape=($(T), $(K)), strides=($(s_k_t), $(1)),
          offsets=(i_t * $(BT) + i_i * $(BC), $(0)), block_shape=($(BC), $(BK)), order=(1, 0))
        p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
          shape=($(T), $(K)), strides=($(s_k_t), $(1)),
          offsets=(i_t * $(BT) + i_i * $(BC), $(0)), block_shape=($(BC), $(BK)), order=(1, 0))
        p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
        p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
        b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat), padding_option="zero")
        for j in range($(0), min($(BC), $(T) - i_t * $(BT) - i_i * $(BC))) {
          b_k = (tl.load(p_k, mask=m_k, other=0)).to(tl.float32)
          b_gk = (tl.load(p_gk, mask=m_k, other=0)).to(tl.float32)
          b_A = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
          $scaled
          tl.store(A + o_A + j, b_A, mask=m_A)
          p_k += $(K)
          p_gk += $(K)
        }
      }).surfaceBody)
end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedIntra

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedSplit
open Lean VeriTile.Triton
syntax "glaSplitSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaSplitSource%[$qTy:tritonDType, $kTy:tritonDType, $gTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scaleFirst do
      Macro.throwError "expected original or scaleFirst"
    let scaled : TSyntax `tritonStmt ←
      if version.getId == `scaleFirst then `(tritonStmt| b_A = tl.where(o_i >= j, $((scale : ℝ)) * b_A, 0.0))
      else `(tritonStmt| b_A = tl.where(o_i >= j, b_A * $((scale : ℝ)), 0.0))
    `(fun (q k g A : RegionName)
        (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat) =>
      ComputeKernel.mk [q,k,g] [A] (triton {
        tl.region q = $qTy, k = $kTy, g = $gTy, A = tl.float32
        i_k = tl.program_id(0)
        i_tc = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_t = i_tc // $(NC)
        i_i = i_tc % $(NC)
        i_j = i_i
        n_bh = tl.num_programs(2)
        if i_t * $(BT) + i_i * $(BC) >= $(T) { return }
        o_i = tl.arange(0, $(BC))
        o_k = i_k * $(BK) + tl.arange(0, $(BK))
        o_A = (i_bh + i_k * n_bh) * $(T) * $(BC) + (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) * $(BC)
        m_k = o_k < $(K)
        m_A = (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) < $(T)
        p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
          shape=($(T), $(K)), strides=($(s_k_t), $(1)),
          offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)), block_shape=($(BC), $(BK)), order=(1, 0))
        p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
          shape=($(T), $(K)), strides=($(s_k_t), $(1)),
          offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)), block_shape=($(BC), $(BK)), order=(1, 0))
        p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
        p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
        b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat), padding_option="zero")
        for j in range($(0), min($(BC), $(T) - i_t * $(BT) - i_i * $(BC))) {
          b_A = tl.zeros([$(BC)], dtype=tl.float32)
          b_k = (tl.load(p_k, mask=m_k, other=0)).to(tl.float32)
          b_gk = (tl.load(p_gk, mask=m_k, other=0)).to(tl.float32)
          b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
          $scaled
          tl.store(A + o_A + j, b_A, mask=m_A)
          p_k += $(K)
          p_gk += $(K)
        }
      }).surfaceBody)
end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedSplit

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedMerge
open Lean VeriTile.Triton
syntax "glaMerge%" "[" ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaMerge%[$version:ident]) => do
    unless version.getId == `original || version.getId == `loadFirst do
      Macro.throwError "expected original or loadFirst"
    let add : TSyntax `tritonStmt ←
      if version.getId == `loadFirst then
        `(tritonStmt| b_A = tl.load(p_A, boundary_check=([0, 1] : List Nat), padding_option="zero") + b_A)
      else `(tritonStmt| b_A += tl.load(p_A, boundary_check=([0, 1] : List Nat), padding_option="zero"))
    `(fun (out_dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) =>
      ComputeKernel.mk [A] [A2] (triton {
        tl.region A = tl.float32, A2 = OUT_DTYPE
        i_t = tl.program_id(0)
        i_c = tl.program_id(1)
        i_bh = tl.program_id(2)
        if i_t * $(BT) + i_c * $(BC) >= $(T) { return }
        n_bh = tl.num_programs(2)
        b_A = tl.zeros([$(BC), $(BC)], dtype=tl.float32)
        for i_k in range($(0), $(NK)) {
          p_A = tl.make_block_ptr(base=A + (i_bh + i_k * n_bh) * $(T) * $(BC),
            shape=($(T), $(BC)), strides=($(BC), $(1)),
            offsets=(i_t * $(BT) + i_c * $(BC), $(0)),
            block_shape=($(BC), $(BC)), order=(1, 0))
          $add
        }
        p_A2 = tl.make_block_ptr(base=A2 + i_bh * $(T) * $(BT),
          shape=($(T), $(BT)), strides=($(BT), $(1)),
          offsets=(i_t * $(BT) + i_c * $(BC), i_c * $(BC)),
          block_shape=($(BC), $(BC)), order=(1, 0))
        tl.store(p_A2, (b_A).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedMerge

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TypedOutput
open Lean VeriTile.Triton
syntax "glaOutputSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaOutputSource%[$qTy:tritonDType, $vTy:tritonDType, $gTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `sumLeft do
      Macro.throwError "expected original or sumLeft"
    let finalAdd : TSyntax `tritonStmt ←
      if version.getId == `sumLeft then `(tritonStmt| b_o = tl.dot(b_A, b_v, allow_tf32=false) + b_o)
      else `(tritonStmt| b_o += tl.dot(b_A, b_v, allow_tf32=false))
    `(fun (dtype out_dtype : FloatDType) (q v g h o A : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV : Nat) =>
      ComputeKernel.mk [q,v,g,h,A] [o] (triton {
        tl.region q = $qTy, v = $vTy, g = $gTy, h = DTYPE,
          o = OUT_DTYPE, A = tl.float32
        i_v = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        m_s = tl.arange(0, $(BT))[:, None] >= tl.arange(0, $(BT))[None, :]
        b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
        for i_k in range(tl.cdiv($(K), $(BK))) {
          p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
            shape=($(T), $(K)), strides=($(s_k_t), $(1)),
            offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
          p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
            shape=($(T), $(K)), strides=($(s_k_t), $(1)),
            offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
            shape=($(K), $(V)), strides=($(s_h_t), $(1)),
            offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_q = (b_q * $((scale : ℝ))).to($qTy:tritonDType)
          b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_qg = (b_q * tl.exp(b_g)).to($qTy:tritonDType)
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
          if i_k >= $(0) {
            b_o += tl.dot(b_qg, (b_h).to($qTy:tritonDType))
          }
        }
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_A = tl.make_block_ptr(base=A + i_bh * $(T) * $(BT),
          shape=($(T), $(BT)), strides=($(BT), $(1)),
          offsets=(i_t * $(BT), $(0)), block_shape=($(BT), $(BT)), order=(1, 0))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_A = tl.load(p_A, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_A = (tl.where(m_s, b_A, 0.0)).to($vTy:tritonDType)
        $finalAdd
        tl.store(p_o, (b_o).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)

end VeriTile.Bench.Repairs.ChunkGlaFwd.TypedOutput

