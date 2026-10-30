import bench.tritonbench_g.attention_fwd_triton1.Sources.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
open VeriTile.Triton
abbrev source := AttentionFwdTriton1.Whole001.TypedSource.repaired
/-- Strength reduction of the four BT=32 temporal pointer offsets. All FP32
operations, casts, boundary checks, history and STORE/IFCOND branches are identical. -/
def shiftedSource
    (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BD NT : Nat) (STORE IFCOND : Bool) :
    ComputeKernel := triton {
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, h = tl.float32, o = tl.float32
  i_bh = (tl.program_id(0)).to(tl.int64)
  b_h = tl.zeros([$(BD), $(BD)], dtype=tl.float32)
  for i in range($(0), $(NT)) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i << $(5), 0), block_shape=($(32), $(BD)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_qh),
      shape=($(BD), $(T)), strides=($(s_qd), $(s_qt)),
      offsets=(0, i << $(5)), block_shape=($(BD), $(32)), order=(0, 1))
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i << $(5), 0), block_shape=($(32), $(BD)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_hh),
      shape=($((NT * BD : Nat)), $(BD)), strides=($(s_ht), 1),
      offsets=(i * $(BD), 0), block_shape=($(BD), $(BD)), order=(1, 0))
    p_o = tl.make_block_ptr(base=o + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i << $(5), 0), block_shape=($(32), $(BD)), order=(1, 0))

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

theorem source_unchanged : source=AttentionFwdTriton1.Whole001.TypedSource.repaired := rfl
#axiomsClean source_unchanged
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
