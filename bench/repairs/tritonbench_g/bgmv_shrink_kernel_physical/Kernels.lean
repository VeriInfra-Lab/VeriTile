import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Types
import VeriTile.Triton.DSL.CallSignature

/-! Original Python bodies and the existing final-multiplication candidate.
Pointer/scalar ABI and constexpr settings are caller metadata. Sentinel return,
all casts, reduction, signed loop and implicit store/atomic conversion remain
as written in Python; no type directive is inserted into a quotation. -/
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Kernels
open VeriTile.Triton
noncomputable section
def signature_fp16_i32 : CallSignature where
  pointers := [("input_ptr",.fp16),("lora_ptr",.fp16),("out_ptr",.outputFloat),("lora_indices",.i32)]
  scalars := [("N",.i32), ("K",.i32), ("xm_stride",.i32), ("xk_stride",.i32), ("l0_stride",.i32), ("lora_k_stride",.i32), ("lora_n_stride",.i32), ("cm_stride",.i32), ("cn_stride",.i32), ("scaling",.fp32)]
  constexpr := ["BLOCK_N", "BLOCK_K", "SPLIT_K"]
  readPorts := ["input_ptr", "lora_ptr", "out_ptr", "lora_indices"]
  writePorts := ["out_ptr"]

def original_fp16_i32 (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (out_dtype : FloatDType) (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call signature_fp16_i32 call_args [N := N, K := K, xm_stride := xm_stride, xk_stride := xk_stride, l0_stride := l0_stride, lora_k_stride := lora_k_stride, lora_n_stride := lora_n_stride, cm_stride := cm_stride, cn_stride := cn_stride] in
  triton {
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

def scalarLeft_fp16_i32 (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (out_dtype : FloatDType) (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call signature_fp16_i32 call_args [N := N, K := K, xm_stride := xm_stride, xk_stride := xk_stride, l0_stride := l0_stride, lora_k_stride := lora_k_stride, lora_n_stride := lora_n_stride, cm_stride := cm_stride, cn_stride := cn_stride] in
  triton {
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

def signature_fp16_i64 : CallSignature where
  pointers := [("input_ptr",.fp16),("lora_ptr",.fp16),("out_ptr",.outputFloat),("lora_indices",.i64)]
  scalars := [("N",.i32), ("K",.i32), ("xm_stride",.i32), ("xk_stride",.i32), ("l0_stride",.i32), ("lora_k_stride",.i32), ("lora_n_stride",.i32), ("cm_stride",.i32), ("cn_stride",.i32), ("scaling",.fp32)]
  constexpr := ["BLOCK_N", "BLOCK_K", "SPLIT_K"]
  readPorts := ["input_ptr", "lora_ptr", "out_ptr", "lora_indices"]
  writePorts := ["out_ptr"]

def original_fp16_i64 (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (out_dtype : FloatDType) (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call signature_fp16_i64 call_args [N := N, K := K, xm_stride := xm_stride, xk_stride := xk_stride, l0_stride := l0_stride, lora_k_stride := lora_k_stride, lora_n_stride := lora_n_stride, cm_stride := cm_stride, cn_stride := cn_stride] in
  triton {
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

def scalarLeft_fp16_i64 (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (out_dtype : FloatDType) (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call signature_fp16_i64 call_args [N := N, K := K, xm_stride := xm_stride, xk_stride := xk_stride, l0_stride := l0_stride, lora_k_stride := lora_k_stride, lora_n_stride := lora_n_stride, cm_stride := cm_stride, cn_stride := cn_stride] in
  triton {
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

def signature_bf16_i32 : CallSignature where
  pointers := [("input_ptr",.bf16),("lora_ptr",.bf16),("out_ptr",.outputFloat),("lora_indices",.i32)]
  scalars := [("N",.i32), ("K",.i32), ("xm_stride",.i32), ("xk_stride",.i32), ("l0_stride",.i32), ("lora_k_stride",.i32), ("lora_n_stride",.i32), ("cm_stride",.i32), ("cn_stride",.i32), ("scaling",.fp32)]
  constexpr := ["BLOCK_N", "BLOCK_K", "SPLIT_K"]
  readPorts := ["input_ptr", "lora_ptr", "out_ptr", "lora_indices"]
  writePorts := ["out_ptr"]

def original_bf16_i32 (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (out_dtype : FloatDType) (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call signature_bf16_i32 call_args [N := N, K := K, xm_stride := xm_stride, xk_stride := xk_stride, l0_stride := l0_stride, lora_k_stride := lora_k_stride, lora_n_stride := lora_n_stride, cm_stride := cm_stride, cn_stride := cn_stride] in
  triton {
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

def scalarLeft_bf16_i32 (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (out_dtype : FloatDType) (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call signature_bf16_i32 call_args [N := N, K := K, xm_stride := xm_stride, xk_stride := xk_stride, l0_stride := l0_stride, lora_k_stride := lora_k_stride, lora_n_stride := lora_n_stride, cm_stride := cm_stride, cn_stride := cn_stride] in
  triton {
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

def signature_bf16_i64 : CallSignature where
  pointers := [("input_ptr",.bf16),("lora_ptr",.bf16),("out_ptr",.outputFloat),("lora_indices",.i64)]
  scalars := [("N",.i32), ("K",.i32), ("xm_stride",.i32), ("xk_stride",.i32), ("l0_stride",.i32), ("lora_k_stride",.i32), ("lora_n_stride",.i32), ("cm_stride",.i32), ("cn_stride",.i32), ("scaling",.fp32)]
  constexpr := ["BLOCK_N", "BLOCK_K", "SPLIT_K"]
  readPorts := ["input_ptr", "lora_ptr", "out_ptr", "lora_indices"]
  writePorts := ["out_ptr"]

def original_bf16_i64 (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (out_dtype : FloatDType) (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call signature_bf16_i64 call_args [N := N, K := K, xm_stride := xm_stride, xk_stride := xk_stride, l0_stride := l0_stride, lora_k_stride := lora_k_stride, lora_n_stride := lora_n_stride, cm_stride := cm_stride, cn_stride := cn_stride] in
  triton {
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

def scalarLeft_bf16_i64 (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (out_dtype : FloatDType) (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call signature_bf16_i64 call_args [N := N, K := K, xm_stride := xm_stride, xk_stride := xk_stride, l0_stride := l0_stride, lora_k_stride := lora_k_stride, lora_n_stride := lora_n_stride, cm_stride := cm_stride, cn_stride := cn_stride] in
  triton {
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

def originalFP16 (h : Call) : ComputeKernel :=
  original_fp16_i64 "X" "W" "Y" ⟨"Indices"⟩ h.outputType
    h.N h.K h.K 1 (h.N*h.K) h.K 1 h.columns 1 h.BN 256 64

def scalarLeftFP16 (h : Call) : ComputeKernel :=
  scalarLeft_fp16_i64 "X" "W" "Y" ⟨"Indices"⟩ h.outputType
    h.N h.K h.K 1 (h.N*h.K) h.K 1 h.columns 1 h.BN 256 64

def originalBF16 (h : Call) : ComputeKernel :=
  original_bf16_i64 "X" "W" "Y" ⟨"Indices"⟩ h.outputType
    h.N h.K h.K 1 (h.N*h.K) h.K 1 h.columns 1 h.BN 256 64

def scalarLeftBF16 (h : Call) : ComputeKernel :=
  scalarLeft_bf16_i64 "X" "W" "Y" ⟨"Indices"⟩ h.outputType
    h.N h.K h.K 1 (h.N*h.K) h.K 1 h.columns 1 h.BN 256 64

/-- The retained mathematical metadata channel is signed Int. Separate i32/i64
quotations record both Python ABIs; native width/validity obligations remain in
the existing caller contracts. The public output-type parameter is retained. -/
def source (h : Call) : ComputeKernel :=
  match h.storage with
  | .fp16 => originalFP16 h
  | .bf16 => originalBF16 h

def scalarLeft (h : Call) : ComputeKernel :=
  match h.storage with
  | .fp16 => scalarLeftFP16 h
  | .bf16 => scalarLeftBF16 h
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Kernels
