import VeriTile.Triton
import VeriTile.Triton.Source
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile.Triton
structure Parameters where
  (B H T K V BT BC BK BV NT : Nat)
  (wBatch wHead wTime wDim : Nat)
  (useSeed storeFinal : Bool)

/-- Full repaired FP32 source. Canonical K/U/history strides come from the
selected wrapper domain; W retains its own four physical strides. -/
def kernel (p : Parameters) : ComputeKernel :=
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
      p_h0 = tl.make_block_ptr(base=initial_state + i_bh * $(p.K) * $(p.V),
        shape=($(p.K), $(p.V)), strides=($(p.V), $(1)),
        offsets=(i_k * $(p.BK), i_v * $(p.BV)),
        block_shape=($(p.BK), $(p.BV)), order=(1, 0))
      b_h = tl.load(p_h0, boundary_check=([0, 1] : List Nat), padding_option="zero").to(tl.float32)
    }
    for i_t in range($(0), $(p.NT), $(1)) {
      p_h = tl.make_block_ptr(base=h + i_bh * $(p.NT*p.K*p.V) + i_t * $(p.K) * $(p.V),
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
      p_ht = tl.make_block_ptr(base=final_state + i_bh * $(p.K) * $(p.V),
        shape=($(p.K), $(p.V)), strides=($(p.V), $(1)),
        offsets=(i_k * $(p.BK), i_v * $(p.BV)),
        block_shape=($(p.BK), $(p.BV)), order=(1, 0))
      tl.store(p_ht, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
    }
  }

def outer (p : Parameters) : Source.DynamicRange :=
  (Source.dynamicRange? (kernel p).surfaceBody "i_t").getD ⟨[],.constNat 0,.constNat 0,.constNat 1,[],[]⟩
def inner (p : Parameters) : Source.DynamicRange :=
  (Source.dynamicRange? (outer p).body "i_c").getD ⟨[],.constNat 0,.constNat 0,.constNat 1,[],[]⟩

theorem outer_parts (p : Parameters) :
    (kernel p).surfaceBody=(outer p).before++
      [.forRange "i_t" 0 p.NT 1 (outer p).body]++(outer p).after := rfl
theorem inner_parts (p : Parameters) :
    (outer p).body=(inner p).before++
      [.forRangeDyn "i_c" (inner p).start (inner p).stop (inner p).stride (inner p).body]++(inner p).after := rfl
#axiomsClean outer_parts
#axiomsClean inner_parts
end VeriTile.Bench.Repairs.ChunkDeltaFwd
