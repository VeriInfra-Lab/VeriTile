import VeriTile.Triton.Source
import VeriTile.Triton.DSL

namespace VeriTile.Bench.Repairs.DequantizeMatmul.TypedSource
open Lean VeriTile.Triton
/-- The complete repaired JIT body; host torch.mm is a separate service.
The source multiplication remains int8 * fp32, with its signed promotion. -/
def kernel (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_scale stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :=
  triton {
  tl.region b_ptr = tl.int8, b_scale_ptr = tl.float32, fpb_ptr = tl.float32
  k_block_idx = tl.program_id(axis=0).to(tl.int64)
  n_block_idx = tl.program_id(axis=1).to(tl.int64)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  offs_n = tl.arange(0, $(BLOCK_SIZE_N))
  b_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_bk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_bn)
  fpb_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_fpbk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_fpbn)
  bs_offs = (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_scale)
  n_mask = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :] < $(N)
  mask = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None] < $(K)) & n_mask
  int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
  tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)
}
end VeriTile.Bench.Repairs.DequantizeMatmul.TypedSource

namespace VeriTile.Bench.Repairs.DequantizeMatmul.KernelSource
open VeriTile.Triton

structure Parameters where
  (b scale output : RegionName)
  (K N BK BN : Nat)

/-- The checked Python passes canonical contiguous strides explicitly, even
for singleton dimensions whose observed Tensor strides may be arbitrary. -/
def original (p : Parameters) : ComputeKernel :=
  TypedSource.kernel p.b p.scale p.output p.K p.N p.N 1 1 p.N 1 p.BN p.BK

/-- The sole candidate change commutes the exact integer N-mask product.
The signed INT8 promotion, FP32 multiply, wider scale mask and stores remain
in source order. This does not claim a Python implementation or speedup. -/
def candidate (p : Parameters) : ComputeKernel :=
  let b_ptr := p.b
  let b_scale_ptr := p.scale
  let fpb_ptr := p.output
  triton {
  tl.region b_ptr = tl.int8, b_scale_ptr = tl.float32, fpb_ptr = tl.float32
  k_block_idx = tl.program_id(axis=0).to(tl.int64)
  n_block_idx = tl.program_id(axis=1).to(tl.int64)
  offs_k = tl.arange(0, $(p.BK))
  offs_n = tl.arange(0, $(p.BN))
  b_offs = (k_block_idx * $(p.BK) + offs_k[:, None]) * $(p.N) +
    (n_block_idx * $(p.BN) + offs_n[None, :]) * $(1)
  fpb_offs = (k_block_idx * $(p.BK) + offs_k[:, None]) * $(p.N) +
    (n_block_idx * $(p.BN) + offs_n[None, :]) * $(1)
  bs_offs = (n_block_idx * $(p.BN) + offs_n[None, :]) * $(1)
  n_mask = $(p.BN) * n_block_idx + offs_n[None, :] < $(p.N)
  mask = (k_block_idx * $(p.BK) + offs_k[:, None] < $(p.K)) & n_mask
  int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
  tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)
  }

def kernel (version : Bool) (p : Parameters) : ComputeKernel :=
  if version then candidate p else original p
end VeriTile.Bench.Repairs.DequantizeMatmul.KernelSource
