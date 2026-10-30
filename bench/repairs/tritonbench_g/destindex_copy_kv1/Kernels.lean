import VeriTile.Triton.DSL.CallSignature
import bench.tritonbench_g.destindex_copy_kv1.Kernels
import Mathlib.Data.Nat.Log
namespace VeriTile.Bench.Repairs.DestindexCopyKv1
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DestindexCopyKv1
structure Layout where
  K : RegionName
  Dest_loc : Region .int
  Out : RegionName
  stride_k_bs : Nat
  stride_k_h : Nat
  stride_k_d : Nat
  stride_o_bs : Nat
  stride_o_h : Nat
  stride_o_d : Nat
  head_num : Nat
  head_dim : Nat
  BLOCK_DMODEL : Nat
  BLOCK_HEAD : Nat

def Layout.source (p : Layout) (input output : TypedSource.Storage) (swapped : Bool) : ComputeKernel :=
  TypedSource.kernel input output swapped p.K p.Dest_loc p.Out p.stride_k_bs p.stride_k_h p.stride_k_d p.stride_o_bs p.stride_o_h p.stride_o_d p.head_num p.head_dim p.BLOCK_DMODEL p.BLOCK_HEAD

/- Preserved source definitions for the existing precision-polymorphic API. -/


/-- Actual invocation types and roles; runtime strides/counts remain runtime. -/
def sourceSignature : CallSignature where
  pointers := [("K",.fp32),("Dest_loc",.i32),("Out",.fp32)]
  scalars := [("stride_k_bs",.integer),("stride_k_h",.integer),("stride_k_d",.integer),("stride_o_bs",.integer),("stride_o_h",.integer),("stride_o_d",.integer),("head_num",.integer),("head_dim",.integer)]
  constexpr := ["BLOCK_DMODEL","BLOCK_HEAD"]
  readPorts := ["K","Dest_loc"]
  writePorts := ["Out"]

/-- Complete actual original Python body, including its function return. -/
def kernel (p : Layout) : ComputeKernel :=
  let K := p.K
  let Dest_loc : RegionName := p.Dest_loc.cast
  let Out := p.Out
  let BLOCK_DMODEL := p.BLOCK_DMODEL
  let BLOCK_HEAD := p.BLOCK_HEAD
  with_triton_call sourceSignature call_args [stride_k_bs := p.stride_k_bs, stride_k_h := p.stride_k_h, stride_k_d := p.stride_k_d, stride_o_bs := p.stride_o_bs, stride_o_h := p.stride_o_h, stride_o_d := p.stride_o_d, head_num := p.head_num, head_dim := p.head_dim] in
  triton {
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    dest_index = tl.load(Dest_loc + cur_index)

    k_ptrs = K + cur_index * stride_k_bs + stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]

    k = tl.load(k_ptrs, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim), other=0.0)
    tl.store(o_ptrs, k, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim))
    return
  }


def block (n : Nat) : Nat := 2 ^ Nat.clog 2 n
structure Config where
  count : Nat
  sourceRows : Nat
  rows : Nat
  heads : Nat
  dims : Nat

def inputRegion : RegionName := ⟨"K"⟩
def metadataRegion : Region .int := ⟨"Dest_loc"⟩
def outputRegion : RegionName := ⟨"Out"⟩
def layout (c : Config) : Layout :=
  ⟨inputRegion, metadataRegion, outputRegion, c.heads*c.dims, c.dims, 1,
    c.heads*c.dims, c.dims, 1, c.heads, c.dims, block c.dims, block c.heads⟩
def flat (c : Config) (row head dim : Nat) : Nat := row*(c.heads*c.dims)+head*c.dims+dim


/-- Complete actual existing index candidate Python body, including its function return. -/
def indexCommutedKernel (p : Layout) : ComputeKernel :=
  let K := p.K
  let Dest_loc : RegionName := p.Dest_loc.cast
  let Out := p.Out
  let BLOCK_DMODEL := p.BLOCK_DMODEL
  let BLOCK_HEAD := p.BLOCK_HEAD
  with_triton_call sourceSignature call_args [stride_k_bs := p.stride_k_bs, stride_k_h := p.stride_k_h, stride_k_d := p.stride_k_d, stride_o_bs := p.stride_o_bs, stride_o_h := p.stride_o_h, stride_o_d := p.stride_o_d, head_num := p.head_num, head_dim := p.head_dim] in
  triton {
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    dest_index = tl.load(Dest_loc + cur_index)

    k_ptrs = K + stride_k_bs * cur_index + stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]

    k = tl.load(k_ptrs, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim), other=0.0)
    tl.store(o_ptrs, k, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim))
    return
  }



/- Older precision-polymorphic comparison instances, kept explicitly separate. -/
namespace Legacy
def kernel (p : Layout) : ComputeKernel := p.source .fp32 .fp32 Bool.false
def indexCommutedKernel (p : Layout) : ComputeKernel := p.source .fp32 .fp32 Bool.true
end Legacy

end VeriTile.Bench.Repairs.DestindexCopyKv1
