import bench.tritonbench_g.destindex_copy_kv2.Sources.Mathematical_Kernels
import bench.tritonbench_g.destindex_copy_kv1.Kernels

namespace VeriTile.Bench.Repairs.DestindexCopyKv2
open VeriTile.Triton
abbrev Layout := VeriTile.Bench.Repairs.DestindexCopyKv1.Layout
abbrev Config := VeriTile.Bench.Repairs.DestindexCopyKv1.Config
abbrev block := VeriTile.Bench.Repairs.DestindexCopyKv1.block
abbrev flat := VeriTile.Bench.Repairs.DestindexCopyKv1.flat

namespace Legacy
def kernel (p : Layout) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.DestindexCopyKv2.TypedSource.kernel .fp32 .fp32 Bool.false
    p.K p.Dest_loc p.Out p.stride_k_bs p.stride_k_h p.stride_k_d
    p.stride_o_bs p.stride_o_h p.stride_o_d p.head_num p.BLOCK_DMODEL p.BLOCK_HEAD

def indexCommutedKernel (p : Layout) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.DestindexCopyKv2.TypedSource.kernel .fp32 .fp32 Bool.true
    p.K p.Dest_loc p.Out p.stride_k_bs p.stride_k_h p.stride_k_d
    p.stride_o_bs p.stride_o_h p.stride_o_d p.head_num p.BLOCK_DMODEL p.BLOCK_HEAD
end Legacy

/-- Complete selected FP32 source. Only the head dimension is masked, as in
the Python source; the public width premise covers every feature lane. -/
def kernel (p : Layout) : ComputeKernel :=
  let K := p.K
  let Dest_loc := p.Dest_loc
  let Out := p.Out
  triton {
    tl.ports inputs=[K, Dest_loc.cast], outputs=[Out]
    tl.region K = tl.float32, Out = tl.float32
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, $(p.BLOCK_HEAD))
    offs_d = tl.arange(0, $(p.BLOCK_DMODEL))
    dest_index = tl.load(Dest_loc + cur_index)
    k_ptrs = K + cur_index * $(p.stride_k_bs) +
      $(p.stride_k_h) * offs_h[:, None] + $(p.stride_k_d) * offs_d[None, :]
    o_ptrs = Out + dest_index * $(p.stride_o_bs) +
      $(p.stride_o_h) * offs_h[:, None] + $(p.stride_o_d) * offs_d[None, :]
    k = tl.load(k_ptrs, mask=offs_h[:, None] < $(p.head_num), other=0.0)
    tl.store(o_ptrs, k.to(tl.float32), mask=offs_h[:, None] < $(p.head_num))
  }

/-- Existing variant: commute only the exact source-row index product. -/
def indexCommutedKernel (p : Layout) : ComputeKernel :=
  let K := p.K
  let Dest_loc := p.Dest_loc
  let Out := p.Out
  triton {
    tl.ports inputs=[K, Dest_loc.cast], outputs=[Out]
    tl.region K = tl.float32, Out = tl.float32
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, $(p.BLOCK_HEAD))
    offs_d = tl.arange(0, $(p.BLOCK_DMODEL))
    dest_index = tl.load(Dest_loc + cur_index)
    k_ptrs = K + $(p.stride_k_bs) * cur_index +
      $(p.stride_k_h) * offs_h[:, None] + $(p.stride_k_d) * offs_d[None, :]
    o_ptrs = Out + dest_index * $(p.stride_o_bs) +
      $(p.stride_o_h) * offs_h[:, None] + $(p.stride_o_d) * offs_d[None, :]
    k = tl.load(k_ptrs, mask=offs_h[:, None] < $(p.head_num), other=0.0)
    tl.store(o_ptrs, k.to(tl.float32), mask=offs_h[:, None] < $(p.head_num))
  }

def layout (c : Config) : Layout :=
  VeriTile.Bench.Repairs.DestindexCopyKv1.layout c
end VeriTile.Bench.Repairs.DestindexCopyKv2
