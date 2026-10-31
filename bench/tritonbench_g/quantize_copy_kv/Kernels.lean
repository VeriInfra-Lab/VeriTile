import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.KernelSource
open Lean VeriTile.Triton

structure Parameters where
  heads : Nat
  width : Nat
  height : Nat

/-! The selected contiguous FP32 source, with FP16 scale storage and signed
int8 output. Runtime strides are the exact values supplied by the canonical
Python wrapper. The candidate changes only one integer address product.
No floating arithmetic, reduction order, cast, mask or input guard changes. -/

/-- Complete unchanged selected Python body and the existing address candidate.
Call storage/stride metadata is supplied outside each raw quotation. -/
def callSignature : CallSignature where
  pointers := [("K",.fp32),("Dest_loc",.i64),("Out",.i8),("Out_scale",.fp16)]
  scalars := [("stride_k_bs",.integer),("stride_k_h",.integer),("stride_k_d",.integer),("stride_dest",.integer),("stride_o_bs",.integer),("stride_o_h",.integer),("stride_o_d",.integer),("stride_os_bs",.integer),("stride_os_h",.integer),("head_num",.integer)]
  constexpr := ["BLOCK_DMODEL","BLOCK_HEAD"]

def originalSource (p : Parameters) : ComputeKernel :=
  let K : RegionName := "input"
  let Dest_loc : RegionName := "destination"
  let Out : RegionName := "output"
  let Out_scale : RegionName := "scale"
  let BLOCK_DMODEL := p.width
  let BLOCK_HEAD := p.height
  let denseStride : Nat := p.heads * p.width
  with_triton_call callSignature call_args [
    stride_k_bs := denseStride, stride_k_h := p.width, stride_k_d := 1, stride_dest := 1, stride_o_bs := denseStride, stride_o_h := p.width, stride_o_d := 1, stride_os_bs := p.heads, stride_os_h := 1, head_num := p.heads] in triton {
    cur_index = tl.program_id(0).to(tl.int64)
    offs_h = tl.arange(0, BLOCK_HEAD).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    dest_index = tl.load(Dest_loc + cur_index * stride_dest)
    src_data = tl.load(K + cur_index * stride_k_bs + offs_h[:, None] * stride_k_h + stride_k_d * offs_d[None, :], mask=offs_h[:, None] < head_num, other=0.0)
    abs_data = tl.abs(src_data)
    maximum = tl.max(abs_data, axis=1)
    data_scale = (maximum / 127.0).to(tl.float16)[:, None]
    denominator = tl.where(maximum[:, None] == 0.0, 1.0, data_scale.to(tl.float32))
    q_src_data = (src_data / denominator).to(tl.int8)
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + stride_os_h * offs_h[:, None]
    tl.store(o_ptrs, q_src_data, mask=offs_h[:, None] < head_num)
    tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < head_num)
  }

def commutedSource (p : Parameters) : ComputeKernel :=
  let K : RegionName := "input"
  let Dest_loc : RegionName := "destination"
  let Out : RegionName := "output"
  let Out_scale : RegionName := "scale"
  let BLOCK_DMODEL := p.width
  let BLOCK_HEAD := p.height
  let denseStride : Nat := p.heads * p.width
  with_triton_call callSignature call_args [
    stride_k_bs := denseStride, stride_k_h := p.width, stride_k_d := 1, stride_dest := 1, stride_o_bs := denseStride, stride_o_h := p.width, stride_o_d := 1, stride_os_bs := p.heads, stride_os_h := 1, head_num := p.heads] in triton {
    cur_index = tl.program_id(0).to(tl.int64)
    offs_h = tl.arange(0, BLOCK_HEAD).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    dest_index = tl.load(Dest_loc + cur_index * stride_dest)
    src_data = tl.load(K + cur_index * stride_k_bs + offs_h[:, None] * stride_k_h + offs_d[None, :] * stride_k_d, mask=offs_h[:, None] < head_num, other=0.0)
    abs_data = tl.abs(src_data)
    maximum = tl.max(abs_data, axis=1)
    data_scale = (maximum / 127.0).to(tl.float16)[:, None]
    denominator = tl.where(maximum[:, None] == 0.0, 1.0, data_scale.to(tl.float32))
    q_src_data = (src_data / denominator).to(tl.int8)
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + stride_os_h * offs_h[:, None]
    tl.store(o_ptrs, q_src_data, mask=offs_h[:, None] < head_num)
    tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < head_num)
  }

def kernel (version : Bool) (p : Parameters) : ComputeKernel :=
  if version then commutedSource p else originalSource p

def originalKernel := kernel Bool.false
def commutedKernel := kernel Bool.true

end VeriTile.Bench.Repairs.QuantizeCopyKv.KernelSource
