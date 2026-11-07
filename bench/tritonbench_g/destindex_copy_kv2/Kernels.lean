import bench.tritonbench_g.destindex_copy_kv1.Kernels
import VeriTile.Triton.DSL.CallSignature

/-! Complete original and existing integer-address candidate bodies. Caller
types, strides and constexpr tile sizes are outside the source quotations. -/
namespace VeriTile.Bench.Repairs.DestindexCopyKv2
open VeriTile.Triton
abbrev Layout := VeriTile.Bench.Repairs.DestindexCopyKv1.Layout
abbrev Config := VeriTile.Bench.Repairs.DestindexCopyKv1.Config
abbrev block := VeriTile.Bench.Repairs.DestindexCopyKv1.block
abbrev flat := VeriTile.Bench.Repairs.DestindexCopyKv1.flat

def layout (c : Config) : Layout :=
  VeriTile.Bench.Repairs.DestindexCopyKv1.layout c

def callSignature : CallSignature where
  pointers := [("K",.fp32),("Dest_loc",.i32),("Out",.fp32)]
  scalars := [("stride_k_bs",.integer),("stride_k_h",.integer),("stride_k_d",.integer),("stride_o_bs",.integer),("stride_o_h",.integer),("stride_o_d",.integer),("head_num",.integer)]
  constexpr := ["BLOCK_DMODEL", "BLOCK_HEAD"]
  readPorts := ["K","Dest_loc"]
  writePorts := ["Out"]

def sourceKernel (K : RegionName) (Dest_loc : Region .int) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d head_num BLOCK_DMODEL BLOCK_HEAD : Nat) : ComputeKernel :=
  with_triton_call callSignature call_args
    [stride_k_bs := stride_k_bs, stride_k_h := stride_k_h, stride_k_d := stride_k_d, stride_o_bs := stride_o_bs, stride_o_h := stride_o_h, stride_o_d := stride_o_d, head_num := head_num] in triton {
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    dest_index = tl.load(Dest_loc + cur_index)

    k_ptrs = K + cur_index * stride_k_bs + stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]

    k = tl.load(k_ptrs, mask=offs_h[:, None] < head_num, other=0.0)
    tl.store(o_ptrs, k, mask=offs_h[:, None] < head_num)
    return
  }

def commutedKernel (K : RegionName) (Dest_loc : Region .int) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d head_num BLOCK_DMODEL BLOCK_HEAD : Nat) : ComputeKernel :=
  with_triton_call callSignature call_args
    [stride_k_bs := stride_k_bs, stride_k_h := stride_k_h, stride_k_d := stride_k_d, stride_o_bs := stride_o_bs, stride_o_h := stride_o_h, stride_o_d := stride_o_d, head_num := head_num] in triton {
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    dest_index = tl.load(Dest_loc + cur_index)

    k_ptrs = K + stride_k_bs * cur_index + stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]

    k = tl.load(k_ptrs, mask=offs_h[:, None] < head_num, other=0.0)
    tl.store(o_ptrs, k, mask=offs_h[:, None] < head_num)
    return
  }

def kernel (p : Layout) : ComputeKernel :=
  sourceKernel p.K p.Dest_loc p.Out p.stride_k_bs p.stride_k_h p.stride_k_d p.stride_o_bs p.stride_o_h p.stride_o_d p.head_num p.BLOCK_DMODEL p.BLOCK_HEAD

def indexCommutedKernel (p : Layout) : ComputeKernel :=
  commutedKernel p.K p.Dest_loc p.Out p.stride_k_bs p.stride_k_h p.stride_k_d p.stride_o_bs p.stride_o_h p.stride_o_d p.head_num p.BLOCK_DMODEL p.BLOCK_HEAD

end VeriTile.Bench.Repairs.DestindexCopyKv2
