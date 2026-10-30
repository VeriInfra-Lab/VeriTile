import bench.tritonbench_g.chunk_delta_fwd.Sources.Kernels

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot
open VeriTile Triton

/-- Original selected static-dot repair: transparent authoritative source reuse. -/
abbrev Parameters := ChunkDeltaFwd.Parameters
abbrev kernel := ChunkDeltaFwd.kernel

theorem shared_source (p : Parameters) : kernel p = ChunkDeltaFwd.kernel p := rfl

/-- Fold K*V at specialization time in all three state/history block-pointer
bases. This removes one runtime integer multiplication at each such site,
including the history setup inside the chunk loop. All FP32 arithmetic, dot
and FMA ordering, masks, branches and output storage remain unchanged. -/
def addressFold (p : Parameters) : ComputeKernel :=
  let USE_INITIAL_STATE := p.useSeed
  let STORE_FINAL_STATE := p.storeFinal
  let k : RegionName := "key"
  let d : RegionName := "weight"
  let v : RegionName := "value"
  let v_new : RegionName := "corrected"
  let h : RegionName := "history"
  let initial_state : RegionName := "initial"
  let final_state : RegionName := "final"
  triton {
    tl.region k = tl.float32, d = tl.float32, v = tl.float32,
      v_new = tl.float32, h = tl.float32, initial_state = tl.float32, final_state = tl.float32
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    b_h = tl.zeros([$(p.BK), $(p.BV)], dtype=tl.float32)
    if USE_INITIAL_STATE {
      p_h0 = tl.make_block_ptr(base=initial_state + i_bh * $(p.K*p.V),
        shape=($(p.K), $(p.V)), strides=($(p.V), $(1)),
        offsets=(i_k * $(p.BK), i_v * $(p.BV)),
        block_shape=($(p.BK), $(p.BV)), order=(1, 0))
      b_h = tl.load(p_h0, boundary_check=([0, 1] : List Nat), padding_option="zero").to(tl.float32)
    }
    for i_t in range($(0), $(p.NT), $(1)) {
      p_h = tl.make_block_ptr(base=h + i_bh * $(p.NT*p.K*p.V) + i_t * $(p.K*p.V),
        shape=($(p.K), $(p.V)), strides=($(p.V), $(1)),
        offsets=(i_k * $(p.BK), i_v * $(p.BV)),
        block_shape=($(p.BK), $(p.BV)), order=(1, 0))
      tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
      b_h_cumsum = tl.zeros([$(p.BK), $(p.BV)], dtype=tl.float32)
      for i_c in range($(0), tl.cdiv($(p.BT), $(p.BC)), $(1)) {
        p_k = tl.make_block_ptr(base=k + i_bh * $(p.T*p.K),
          shape=($(p.K), $(p.T)), strides=($(1), $(p.K)),
          offsets=(i_k * $(p.BK), i_t * $(p.BT) + i_c * $(p.BC)),
          block_shape=($(p.BK), $(p.BC)), order=(0, 1))
        p_d = tl.make_block_ptr(base=d + (i_bh // $(p.H)) * $(p.wBatch) + (i_bh % $(p.H)) * $(p.wHead),
          shape=($(p.T), $(p.K)), strides=($(p.wTime), $(p.wDim)),
          offsets=(i_t * $(p.BT) + i_c * $(p.BC), i_k * $(p.BK)),
          block_shape=($(p.BC), $(p.BK)), order=(1, 0))
        p_v = tl.make_block_ptr(base=v + i_bh * $(p.T*p.V),
          shape=($(p.T), $(p.V)), strides=($(p.V), $(1)),
          offsets=(i_t * $(p.BT) + i_c * $(p.BC), i_v * $(p.BV)),
          block_shape=($(p.BC), $(p.BV)), order=(1, 0))
        p_v_new = tl.make_block_ptr(base=v_new + i_bh * $(p.T*p.V),
          shape=($(p.T), $(p.V)), strides=($(p.V), $(1)),
          offsets=(i_t * $(p.BT) + i_c * $(p.BC), i_v * $(p.BV)),
          block_shape=($(p.BC), $(p.BV)), order=(1, 0))
        b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_d = tl.load(p_d, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_v -= tl.dot(b_d, (b_h).to(tl.float32), allow_tf32=false)
        tl.store(p_v_new, (b_v).to(tl.float32), boundary_check=([0, 1] : List Nat))
        b_h_cumsum += tl.dot(b_k, (b_v).to(tl.float32), allow_tf32=false)
      }
      b_h += b_h_cumsum
    }
    if STORE_FINAL_STATE {
      p_ht = tl.make_block_ptr(base=final_state + i_bh * $(p.K*p.V),
        shape=($(p.K), $(p.V)), strides=($(p.V), $(1)),
        offsets=(i_k * $(p.BK), i_v * $(p.BV)),
        block_shape=($(p.BK), $(p.BV)), order=(1, 0))
      tl.store(p_ht, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
    }
  }


#axiomsClean shared_source
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot
