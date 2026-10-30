import VeriTile.Triton.DSL

/-! Complete repaired source. FP32 alias casts remain explicit. The host
supplies canonical dense geometry; masks repair temporal tails and H uses
its own unit inner stride. Source/host opcode semantics and compiler binding
are documented separately; this transcription alone is not a translation proof. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001.TypedSource
open VeriTile.Triton

def repaired
    (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool) :
    ComputeKernel := triton {
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, h = tl.float32, o = tl.float32
  i_bh = (tl.program_id(0)).to(tl.int64)
  b_h = tl.zeros([$(BD), $(BD)], dtype=tl.float32)
  for i in range($(0), $(NT)) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_qh),
      shape=($(BD), $(T)), strides=($(s_qd), $(s_qt)),
      offsets=(0, i * $(BT)), block_shape=($(BD), $(BT)), order=(0, 1))
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_hh),
      shape=($((NT * BD : Nat)), $(BD)), strides=($(s_ht), 1),
      offsets=(i * $(BD), 0), block_shape=($(BD), $(BD)), order=(1, 0))
    p_o = tl.make_block_ptr(base=o + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))

    if STORE {
      tl.store(p_h, (b_h).to(tl.float32))
    }
    b_q = tl.load(p_q, boundary_check=([0] : List Nat), padding_option="zero")
    b_q = (b_q * $((scale : ℝ))).to(tl.float32)
    b_k = tl.load(p_k, boundary_check=([1] : List Nat), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=([0] : List Nat), padding_option="zero")

    b_s = tl.dot(b_q, b_k, allow_tf32=false)
    b_o = tl.dot((b_s).to(tl.float32), b_v, allow_tf32=false)
    if IFCOND {
      if i == $(0) {
        b_h = tl.dot(b_k, b_v, allow_tf32=false)
      } else {
        b_o += tl.dot(b_q, (b_h).to(tl.float32), allow_tf32=false)
        b_h += tl.dot(b_k, b_v, allow_tf32=false)
      }
    } else {
      b_o += tl.dot(b_q, (b_h).to(tl.float32), allow_tf32=false)
      b_h += tl.dot(b_k, b_v, allow_tf32=false)
    }

    tl.store(p_o, (b_o).to(tl.float32), boundary_check=([0] : List Nat))
  }
}


end VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001.TypedSource
