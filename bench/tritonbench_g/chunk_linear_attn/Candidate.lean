import bench.tritonbench_g.chunk_linear_attn.Kernels

namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile.Triton

/-! Actual loop-invariant address candidate. Q and K share the contiguous
head stride. Compute that head offset once per output program and reuse it in
both block pointers on every key iteration. All floating operations, masks,
casts, loop iterations and store geometry remain the selected source. -/
def forwardOHoisted (p : Parameters) (scale : ℝ) : ComputeKernel :=
  let T := p.T
  let K := p.K
  let V := p.V
  let BT := p.BT
  let BK := p.BK
  let BV := p.BV
  let NT := p.NT
  let s_qk_h : Nat := p.T*p.K
  let s_qk_t : Nat := p.K
  let s_qk_d : Nat := 1
  let s_vo_h : Nat := p.T*p.V
  let s_vo_t : Nat := p.V
  let s_vo_d : Nat := 1
  let s_h_h : Nat := p.NT*p.K*p.V
  let s_h_t : Nat := p.V
  let q : RegionName := "query"
  let k : RegionName := "key"
  let v : RegionName := "value"
  let h : RegionName := "history"
  let o : RegionName := "output"
  triton {
        tl.region q = tl.float32, k = tl.float32, h = tl.float32, v = tl.float32, o = tl.float32
        i_v = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        o_i = tl.arange(0, $(BT))
        m_s = o_i[:, None] >= o_i[None, :]
        b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
        b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
        qk_head = i_bh * $(s_qk_h)
        for i_k in range(tl.cdiv($(K), $(BK))) {
          p_q = tl.make_block_ptr(base=q + qk_head, shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
          p_k = tl.make_block_ptr(base=k + qk_head, shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_o += tl.dot(b_q, b_h, allow_tf32=False)
          b_s += tl.dot(b_q, b_k, allow_tf32=False)
        }
        b_s = tl.where(m_s, b_s, 0)
        p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
        p_o = tl.make_block_ptr(base=o + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_o = (b_o + tl.dot((b_s).to(tl.float32), b_v, allow_tf32=False)) * $((scale : ℝ))
        tl.store(p_o, (b_o).to(tl.float32), boundary_check=([0, 1] : List Nat))
      }

end VeriTile.Bench.Repairs.ChunkLinearAttn
