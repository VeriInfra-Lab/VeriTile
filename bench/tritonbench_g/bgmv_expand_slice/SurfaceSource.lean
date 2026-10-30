import bench.tritonbench_g.bgmv_expand_slice.Definitions
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Surface
open VeriTile.Triton
/-- Complete primary JIT quotation. Signed casts, the early return, signed
cdiv/range and implicit output conversion are retained in the shared DSL. -/
def source (h : Call) : ComputeKernel :=
  let X : RegionName := "X"
  let W : RegionName := "W"
  let Y : RegionName := "Y"
  let lora_indices : Region .int := ⟨"Indices"⟩
  let EVEN_K : Bool := h.even
  let ADD_INPUTS : Bool := h.add
  triton {
  tl.region X = tl.float32, W = tl.float16, Y = tl.float32
  pid_sn = tl.program_id(axis=0).to(tl.int64)
  cur_batch = tl.program_id(axis=1).to(tl.int64)
  lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
  if lora_index == $((-1 : Int)) { return }
  offset_k = tl.arange(0, $(h.BK)).to(tl.int64)
  offset_n = tl.arange(0, 256).to(tl.int64)
  if EVEN_K {
    tiled_a = tl.load(X + cur_batch * $(h.K) + offset_k)
  } else {
    tiled_a = tl.load(X + cur_batch * $(h.K) + offset_k, mask=offset_k < $(h.K), other=0)
  }
  split_n_length = ($((h.N : Int)) + $((h.N : Int)) - $((1 : Int))) // $((h.N : Int))
  tiled_a = (tiled_a).to(tl.float16)
  b_ptr = W + $(h.N*h.K) * lora_index + pid_sn * split_n_length * $(h.K)
  c_ptr = Y + cur_batch * $(h.columns) + pid_sn * split_n_length + $(h.slice)
  for n in range(($((0 : Int))).to(tl.int64), split_n_length, 256) {
    current_n = n + offset_n
    b_ptr_mask = (current_n[:, None] < split_n_length) & (offset_k[None, :] < $(h.K))
    c_mask = current_n < split_n_length
    tiled_b = tl.load(b_ptr + current_n[:, None] * $(h.K) + offset_k[None, :], mask=b_ptr_mask, other=0.0)
    if ADD_INPUTS {
      tiled_out = tl.load(c_ptr + current_n, mask=c_mask)
      accumulator = (tl.sum(tiled_a * tiled_b, 1) + tiled_out).to(tl.float32)
    } else {
      accumulator = (tl.sum(tiled_a * tiled_b, 1)).to(tl.float32)
    }
    tl.store(c_ptr + current_n, accumulator, mask=c_mask)
  }
}
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Surface
