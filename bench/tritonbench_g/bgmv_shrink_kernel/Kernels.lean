import bench.tritonbench_g.bgmv_shrink_kernel.Target

/-! Actual complete signed split-K sources. Runtime scaling is an explicit
parameter, supplied by IO.invocation. The scalar-left candidate only exchanges
the operands of the final FP32 multiplication. Storage and atomic conversion,
all 64 split programs, loop order and masks remain unchanged. -/
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Kernels
open VeriTile.Triton

def originalFP16 (h : Call) : ComputeKernel :=
  let X : RegionName := "X"
  let W : RegionName := "W"
  let Y : RegionName := "Y"
  let lora_indices : Region .int := ⟨"Indices"⟩
  let out_dtype := h.outputType
  triton {
    tl.ports inputs=[X,W,Y,Region.cast lora_indices], outputs=[Y]
    tl.region X = tl.float16, W = tl.float16, Y = OUT_DTYPE
    tl.param scaling : tl.float32
    pid_sk = tl.program_id(axis=0).to(tl.int64)
    cur_batch = tl.program_id(axis=1).to(tl.int64)
    lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
    if lora_index != $((-1 : Int)) {
      offset_n = tl.arange(0, $(h.BN)).to(tl.int64)
      offset_k = tl.arange(0, $(256)).to(tl.int64) + pid_sk * $(256)
      a_ptr = X + cur_batch * $(h.K)
      b_ptr = W + $(h.N*h.K) * lora_index
      accumulator = tl.zeros([$(h.BN)], dtype=tl.float32)
      for k in range(($((0 : Int))).to(tl.int64), ($((h.K : Int))).to(tl.int64), $(16384)) {
        current_k = k + offset_k
        current_k_c = tl.max_contiguous(current_k, $(256))
        tiled_a = tl.load(a_ptr + current_k_c, mask=current_k < $(h.K), other=0.0)
        b_ptr_mask = (offset_n[:, None] < $(h.N)) & (current_k[None, :] < $(h.K))
        tiled_b = tl.load(
          b_ptr + offset_n[:, None] * $(h.K) +
            current_k[None, :] * $(1),
          mask=b_ptr_mask, other=0.0)
        accumulator += tl.sum(tiled_a * tiled_b, 1)
      }
      accumulator = accumulator * scaling
      offset_cn = tl.arange(0, $(h.BN)).to(tl.int64)
      c_ptr = Y + cur_batch * $(h.columns) + offset_cn * $(1)
      c_mask = offset_cn < $(h.N)
      if $(64) == $(1) {
        tl.store(c_ptr, (accumulator).to(OUT_DTYPE), mask=c_mask)
      } else {
        tl.atomic_add(c_ptr, (accumulator).to(OUT_DTYPE), mask=c_mask)
      }
    }
  }

def scalarLeftFP16 (h : Call) : ComputeKernel :=
  let X : RegionName := "X"
  let W : RegionName := "W"
  let Y : RegionName := "Y"
  let lora_indices : Region .int := ⟨"Indices"⟩
  let out_dtype := h.outputType
  triton {
    tl.ports inputs=[X,W,Y,Region.cast lora_indices], outputs=[Y]
    tl.region X = tl.float16, W = tl.float16, Y = OUT_DTYPE
    tl.param scaling : tl.float32
    pid_sk = tl.program_id(axis=0).to(tl.int64)
    cur_batch = tl.program_id(axis=1).to(tl.int64)
    lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
    if lora_index != $((-1 : Int)) {
      offset_n = tl.arange(0, $(h.BN)).to(tl.int64)
      offset_k = tl.arange(0, $(256)).to(tl.int64) + pid_sk * $(256)
      a_ptr = X + cur_batch * $(h.K)
      b_ptr = W + $(h.N*h.K) * lora_index
      accumulator = tl.zeros([$(h.BN)], dtype=tl.float32)
      for k in range(($((0 : Int))).to(tl.int64), ($((h.K : Int))).to(tl.int64), $(16384)) {
        current_k = k + offset_k
        current_k_c = tl.max_contiguous(current_k, $(256))
        tiled_a = tl.load(a_ptr + current_k_c, mask=current_k < $(h.K), other=0.0)
        b_ptr_mask = (offset_n[:, None] < $(h.N)) & (current_k[None, :] < $(h.K))
        tiled_b = tl.load(
          b_ptr + offset_n[:, None] * $(h.K) +
            current_k[None, :] * $(1),
          mask=b_ptr_mask, other=0.0)
        accumulator += tl.sum(tiled_a * tiled_b, 1)
      }
      accumulator = scaling * accumulator
      offset_cn = tl.arange(0, $(h.BN)).to(tl.int64)
      c_ptr = Y + cur_batch * $(h.columns) + offset_cn * $(1)
      c_mask = offset_cn < $(h.N)
      if $(64) == $(1) {
        tl.store(c_ptr, (accumulator).to(OUT_DTYPE), mask=c_mask)
      } else {
        tl.atomic_add(c_ptr, (accumulator).to(OUT_DTYPE), mask=c_mask)
      }
    }
  }

def originalBF16 (h : Call) : ComputeKernel :=
  let X : RegionName := "X"
  let W : RegionName := "W"
  let Y : RegionName := "Y"
  let lora_indices : Region .int := ⟨"Indices"⟩
  let out_dtype := h.outputType
  triton {
    tl.ports inputs=[X,W,Y,Region.cast lora_indices], outputs=[Y]
    tl.region X = tl.bfloat16, W = tl.bfloat16, Y = OUT_DTYPE
    tl.param scaling : tl.float32
    pid_sk = tl.program_id(axis=0).to(tl.int64)
    cur_batch = tl.program_id(axis=1).to(tl.int64)
    lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
    if lora_index != $((-1 : Int)) {
      offset_n = tl.arange(0, $(h.BN)).to(tl.int64)
      offset_k = tl.arange(0, $(256)).to(tl.int64) + pid_sk * $(256)
      a_ptr = X + cur_batch * $(h.K)
      b_ptr = W + $(h.N*h.K) * lora_index
      accumulator = tl.zeros([$(h.BN)], dtype=tl.float32)
      for k in range(($((0 : Int))).to(tl.int64), ($((h.K : Int))).to(tl.int64), $(16384)) {
        current_k = k + offset_k
        current_k_c = tl.max_contiguous(current_k, $(256))
        tiled_a = tl.load(a_ptr + current_k_c, mask=current_k < $(h.K), other=0.0)
        b_ptr_mask = (offset_n[:, None] < $(h.N)) & (current_k[None, :] < $(h.K))
        tiled_b = tl.load(
          b_ptr + offset_n[:, None] * $(h.K) +
            current_k[None, :] * $(1),
          mask=b_ptr_mask, other=0.0)
        accumulator += tl.sum(tiled_a * tiled_b, 1)
      }
      accumulator = accumulator * scaling
      offset_cn = tl.arange(0, $(h.BN)).to(tl.int64)
      c_ptr = Y + cur_batch * $(h.columns) + offset_cn * $(1)
      c_mask = offset_cn < $(h.N)
      if $(64) == $(1) {
        tl.store(c_ptr, (accumulator).to(OUT_DTYPE), mask=c_mask)
      } else {
        tl.atomic_add(c_ptr, (accumulator).to(OUT_DTYPE), mask=c_mask)
      }
    }
  }

def scalarLeftBF16 (h : Call) : ComputeKernel :=
  let X : RegionName := "X"
  let W : RegionName := "W"
  let Y : RegionName := "Y"
  let lora_indices : Region .int := ⟨"Indices"⟩
  let out_dtype := h.outputType
  triton {
    tl.ports inputs=[X,W,Y,Region.cast lora_indices], outputs=[Y]
    tl.region X = tl.bfloat16, W = tl.bfloat16, Y = OUT_DTYPE
    tl.param scaling : tl.float32
    pid_sk = tl.program_id(axis=0).to(tl.int64)
    cur_batch = tl.program_id(axis=1).to(tl.int64)
    lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
    if lora_index != $((-1 : Int)) {
      offset_n = tl.arange(0, $(h.BN)).to(tl.int64)
      offset_k = tl.arange(0, $(256)).to(tl.int64) + pid_sk * $(256)
      a_ptr = X + cur_batch * $(h.K)
      b_ptr = W + $(h.N*h.K) * lora_index
      accumulator = tl.zeros([$(h.BN)], dtype=tl.float32)
      for k in range(($((0 : Int))).to(tl.int64), ($((h.K : Int))).to(tl.int64), $(16384)) {
        current_k = k + offset_k
        current_k_c = tl.max_contiguous(current_k, $(256))
        tiled_a = tl.load(a_ptr + current_k_c, mask=current_k < $(h.K), other=0.0)
        b_ptr_mask = (offset_n[:, None] < $(h.N)) & (current_k[None, :] < $(h.K))
        tiled_b = tl.load(
          b_ptr + offset_n[:, None] * $(h.K) +
            current_k[None, :] * $(1),
          mask=b_ptr_mask, other=0.0)
        accumulator += tl.sum(tiled_a * tiled_b, 1)
      }
      accumulator = scaling * accumulator
      offset_cn = tl.arange(0, $(h.BN)).to(tl.int64)
      c_ptr = Y + cur_batch * $(h.columns) + offset_cn * $(1)
      c_mask = offset_cn < $(h.N)
      if $(64) == $(1) {
        tl.store(c_ptr, (accumulator).to(OUT_DTYPE), mask=c_mask)
      } else {
        tl.atomic_add(c_ptr, (accumulator).to(OUT_DTYPE), mask=c_mask)
      }
    }
  }

def source (h : Call) : ComputeKernel :=
  match h.storage with
  | .fp16 => originalFP16 h
  | .bf16 => originalBF16 h

def scalarLeft (h : Call) : ComputeKernel :=
  match h.storage with
  | .fp16 => scalarLeftFP16 h
  | .bf16 => scalarLeftBF16 h
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Kernels
