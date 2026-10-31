import VeriTile.Triton.DSL
import Mathlib.Data.Nat.Log
namespace VeriTile.Bench.Repairs.DestindexCopyKv1
open VeriTile.Triton
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

/-- Complete selected FP32 source. The scalar destination load is unmasked;
both payload dimensions retain their original masks. -/
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
    k = tl.load(k_ptrs,
      mask=(offs_h[:, None] < $(p.head_num)) & (offs_d[None, :] < $(p.head_dim)),
      other=0.0)
    tl.store(o_ptrs, k.to(tl.float32),
      mask=(offs_h[:, None] < $(p.head_num)) & (offs_d[None, :] < $(p.head_dim)))
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
    k = tl.load(k_ptrs,
      mask=(offs_h[:, None] < $(p.head_num)) & (offs_d[None, :] < $(p.head_dim)),
      other=0.0)
    tl.store(o_ptrs, k.to(tl.float32),
      mask=(offs_h[:, None] < $(p.head_num)) & (offs_d[None, :] < $(p.head_dim)))
  }

end VeriTile.Bench.Repairs.DestindexCopyKv1
