import VeriTile.Triton.Source
import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

namespace VeriTile.Bench.Repairs.DequantizeMatmul.TypedSource
open Lean VeriTile.Triton

/-- Call-side types for the complete source. Runtime scalar widths remain
mathematical integer channels; existing native index bounds are proved in IO. -/
def sourceSignature : CallSignature where
  pointers := [("b_ptr",.i8),("b_scale_ptr",.fp32),("fpb_ptr",.fp32)]
  scalars := [("K",.integer),("N",.integer),("stride_bk",.integer),
    ("stride_bn",.integer),("stride_scale",.integer),
    ("stride_fpbk",.integer),("stride_fpbn",.integer)]
  constexpr := ["BLOCK_SIZE_N","BLOCK_SIZE_K"]
  readPorts := ["b_ptr","b_scale_ptr"]
  writePorts := ["fpb_ptr"]

def kernel (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_scale stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) : ComputeKernel :=
  let K_value : Nat := K
  let N_value : Nat := N
  let stride_bk_value : Nat := stride_bk
  let stride_bn_value : Nat := stride_bn
  let stride_scale_value : Nat := stride_scale
  let stride_fpbk_value : Nat := stride_fpbk
  let stride_fpbn_value : Nat := stride_fpbn
  with_triton_call sourceSignature call_args
    [K := K_value, N := N_value, stride_bk := stride_bk_value, stride_bn := stride_bn_value, stride_scale := stride_scale_value, stride_fpbk := stride_fpbk_value, stride_fpbn := stride_fpbn_value] in
  triton {
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    k_block_idx = tl.program_id(axis=0).to(tl.int64)
    n_block_idx = tl.program_id(axis=1).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    b_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_bk + \
        (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_bn
    fpb_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_fpbk + \
        (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_fpbn
    bs_offs = (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_scale
    n_mask = n_block_idx * BLOCK_SIZE_N + offs_n[None, :] < N
    mask = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None] < K) & n_mask
    int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
    tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)
  }

def indexCommutedKernel (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_scale stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) : ComputeKernel :=
  let K_value : Nat := K
  let N_value : Nat := N
  let stride_bk_value : Nat := stride_bk
  let stride_bn_value : Nat := stride_bn
  let stride_scale_value : Nat := stride_scale
  let stride_fpbk_value : Nat := stride_fpbk
  let stride_fpbn_value : Nat := stride_fpbn
  with_triton_call sourceSignature call_args
    [K := K_value, N := N_value, stride_bk := stride_bk_value, stride_bn := stride_bn_value, stride_scale := stride_scale_value, stride_fpbk := stride_fpbk_value, stride_fpbn := stride_fpbn_value] in
  triton {
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    k_block_idx = tl.program_id(axis=0).to(tl.int64)
    n_block_idx = tl.program_id(axis=1).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    b_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_bk + \
        (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_bn
    fpb_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_fpbk + \
        (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_fpbn
    bs_offs = (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_scale
    n_mask = BLOCK_SIZE_N * n_block_idx + offs_n[None, :] < N
    mask = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None] < K) & n_mask
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

/-- Canonical contiguous call used by the repaired wrapper. -/
def original (p : Parameters) : ComputeKernel :=
  TypedSource.kernel p.b p.scale p.output p.K p.N p.N 1 1 p.N 1 p.BN p.BK

/-- Existing integer-mask candidate, retaining all source floating operations. -/
def candidate (p : Parameters) : ComputeKernel :=
  TypedSource.indexCommutedKernel p.b p.scale p.output p.K p.N p.N 1 1 p.N 1 p.BN p.BK

def kernel (version : Bool) (p : Parameters) : ComputeKernel :=
  if version then candidate p else original p
end VeriTile.Bench.Repairs.DequantizeMatmul.KernelSource
