import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.Support.Types
import VeriTile.Triton.DSL.ImplicitAtomicOperands
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Pasted
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000

inductive IndexStorage where
  | i32 | i64
  deriving DecidableEq

def callSignature (input_dtype output_dtype : FloatDType) (index_dtype : IndexStorage) : CallSignature where
  pointers := [("lora_indices",match index_dtype with | .i32 => .i32 | .i64 => .i64)]
  floatingPointers := [("input_ptr",input_dtype),("lora_ptr",input_dtype),("out_ptr",output_dtype)]
  scalars := [("N",.integer),("K",.integer),("xm_stride",.integer),("xk_stride",.integer),("l0_stride",.integer),("lora_k_stride",.integer),("lora_n_stride",.integer),("cm_stride",.integer),("cn_stride",.integer),("scaling",.fp32)]
  constexpr := ["BLOCK_N","BLOCK_K","SPLIT_K"]
  readPorts := ["input_ptr","lora_ptr","out_ptr","lora_indices"]
  writePorts := ["out_ptr"]

def originalKernel (input_dtype output_dtype : FloatDType) (index_dtype : IndexStorage)
    (input_ptr lora_ptr out_ptr lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride : Nat) (BLOCK_N BLOCK_K SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call (callSignature input_dtype output_dtype index_dtype)
    call_args [N := N,K := K,xm_stride := xm_stride,xk_stride := xk_stride,l0_stride := l0_stride,lora_k_stride := lora_k_stride,lora_n_stride := lora_n_stride,cm_stride := cm_stride,cn_stride := cn_stride] in triton {
    pid_sk = tl.program_id(axis=0).to(tl.int64)
    cur_batch = tl.program_id(axis=1).to(tl.int64)
    lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
    if lora_index == -1:
        return

    offset_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offset_k = tl.arange(0, BLOCK_K).to(tl.int64) + pid_sk * BLOCK_K
    a_ptr = input_ptr + cur_batch * xm_stride
    b_ptr = lora_ptr + l0_stride * lora_index
    accumulator = tl.zeros((BLOCK_N, ), dtype=tl.float32)
    for k in range(tl.cast(0, tl.int64), tl.cast(K, tl.int64), BLOCK_K * SPLIT_K):
        current_k = k + offset_k
        current_k_c = tl.max_contiguous(current_k, BLOCK_K)
        tiled_a = tl.load(
            a_ptr + current_k_c,
            mask=current_k < K,
            other=0.0,
        )
        b_ptr_mask = (offset_n[:, None] < N) & (current_k[None, :] < K)

        tiled_b = tl.load(
            b_ptr + offset_n[:, None] * lora_k_stride +
            current_k[None, :] * lora_n_stride,
            mask=b_ptr_mask,
            other=0.0,
        )

        accumulator += tl.sum(tiled_a * tiled_b, 1)
    accumulator *= scaling
    offset_cn = tl.arange(0, BLOCK_N).to(tl.int64)
    c_ptr = out_ptr + cur_batch * cm_stride + offset_cn * cn_stride
    c_mask = offset_cn < N
    if SPLIT_K == 1:
        tl.store(c_ptr, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, accumulator, mask=c_mask)
  }

def candidateKernel (input_dtype output_dtype : FloatDType) (index_dtype : IndexStorage)
    (input_ptr lora_ptr out_ptr lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride : Nat) (BLOCK_N BLOCK_K SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call (callSignature input_dtype output_dtype index_dtype)
    call_args [N := N,K := K,xm_stride := xm_stride,xk_stride := xk_stride,l0_stride := l0_stride,lora_k_stride := lora_k_stride,lora_n_stride := lora_n_stride,cm_stride := cm_stride,cn_stride := cn_stride] in triton {
    pid_sk = tl.program_id(axis=0).to(tl.int64)
    cur_batch = tl.program_id(axis=1).to(tl.int64)
    lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
    if lora_index == -1:
        return

    offset_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offset_k = tl.arange(0, BLOCK_K).to(tl.int64) + pid_sk * BLOCK_K
    a_ptr = input_ptr + cur_batch * xm_stride
    b_ptr = lora_ptr + l0_stride * lora_index
    accumulator = tl.zeros((BLOCK_N, ), dtype=tl.float32)
    for k in range(tl.cast(0, tl.int64), tl.cast(K, tl.int64), BLOCK_K * SPLIT_K):
        current_k = k + offset_k
        current_k_c = tl.max_contiguous(current_k, BLOCK_K)
        tiled_a = tl.load(
            a_ptr + current_k_c,
            mask=current_k < K,
            other=0.0,
        )
        b_ptr_mask = (offset_n[:, None] < N) & (current_k[None, :] < K)

        tiled_b = tl.load(
            b_ptr + offset_n[:, None] * lora_k_stride +
            current_k[None, :] * lora_n_stride,
            mask=b_ptr_mask,
            other=0.0,
        )

        accumulator += tl.sum(tiled_a * tiled_b, 1)
    accumulator = scaling * accumulator
    offset_cn = tl.arange(0, BLOCK_N).to(tl.int64)
    c_ptr = out_ptr + cur_batch * cm_stride + offset_cn * cn_stride
    c_mask = offset_cn < N
    if SPLIT_K == 1:
        tl.store(c_ptr, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, accumulator, mask=c_mask)
  }

end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Pasted

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Kernels
open VeriTile.Triton

def originalFP16 (h : Call) : ComputeKernel :=
  Pasted.originalKernel .fp16 h.outputType .i64 "X" "W" "Y" "Indices"
    h.N h.K h.K 1 (h.N*h.K) h.K 1 h.columns 1 h.BN 256 64

def scalarLeftFP16 (h : Call) : ComputeKernel :=
  Pasted.candidateKernel .fp16 h.outputType .i64 "X" "W" "Y" "Indices"
    h.N h.K h.K 1 (h.N*h.K) h.K 1 h.columns 1 h.BN 256 64

def originalBF16 (h : Call) : ComputeKernel :=
  Pasted.originalKernel .bf16 h.outputType .i64 "X" "W" "Y" "Indices"
    h.N h.K h.K 1 (h.N*h.K) h.K 1 h.columns 1 h.BN 256 64

def scalarLeftBF16 (h : Call) : ComputeKernel :=
  Pasted.candidateKernel .bf16 h.outputType .i64 "X" "W" "Y" "Indices"
    h.N h.K h.K 1 (h.N*h.K) h.K 1 h.columns 1 h.BN 256 64

def source (h : Call) : ComputeKernel :=
  match h.storage with
  | .fp16 => originalFP16 h
  | .bf16 => originalBF16 h

def scalarLeft (h : Call) : ComputeKernel :=
  match h.storage with
  | .fp16 => scalarLeftFP16 h
  | .bf16 => scalarLeftBF16 h
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Kernels
