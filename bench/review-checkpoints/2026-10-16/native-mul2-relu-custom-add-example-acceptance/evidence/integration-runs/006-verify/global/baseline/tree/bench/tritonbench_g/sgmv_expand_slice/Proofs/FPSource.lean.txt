import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false
inductive Storage where
  | half | bfloat
  deriving DecidableEq, Repr
inductive Output where
  | half | bfloat | single
  deriving DecidableEq, Repr
def storageType : Storage → FloatDType
  | .half => .fp16 | .bfloat => .bf16
def outputType : Output → FloatDType
  | .half => .fp16 | .bfloat => .bf16 | .single => .fp32

structure Parameters where
  (input weight output : RegionName)
  (starts lengths adapters : Region .nat)
  (columns inner inputRowStride inputInnerStride adapterStride weightColumnStride weightInnerStride
    outputRowStride outputColumnStride sliceOffset blockRows blockColumns blockInner : Nat)

def halfTyped (out_dtype : FloatDType)
    (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  tl.region input_ptr = tl.float16, lora_ptr = tl.float16, out_ptr = OUT_DTYPE
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  cur_batch = tl.program_id(2)
  M = tl.load(seq_lens + cur_batch)
  cur_seq_start = tl.load(b_seq_start_loc + cur_batch)
  lora_index = tl.load(lora_indices + cur_batch)
  offset_m = tl.arange(0, $(BLOCK_M)) + pid_m * $(BLOCK_M)
  offset_n = tl.arange(0, $(BLOCK_N)) + pid_n * $(BLOCK_N)
  offset_k = tl.arange(0, $(BLOCK_K))
  ram = offset_m % M
  rbn = offset_n % $(N)
  a_ptr = input_ptr + cur_seq_start * $(xm_stride) +
    ram[:, None] * $(xm_stride) + offset_k[None, :] * $(xk_stride)
  b_ptr = lora_ptr + $(l0_stride) * lora_index +
    offset_k[:, None] * $(lora_n_stride) + rbn[None, :] * $(lora_k_stride)
  accumulator = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  for kk in range($(0), $(K), $(BLOCK_K)) {
    tiled_a = tl.load(a_ptr)
    tiled_b = tl.load(b_ptr)
    accumulator += tl.dot(tiled_a, tiled_b)
    a_ptr += $(BLOCK_K) * $(xk_stride)
    b_ptr += $(BLOCK_K) * $(lora_n_stride)
  }
  offset_cm = cur_seq_start + offset_m
  offset_cn = offset_n + $(slice_offset)
  c_ptr = out_ptr + offset_cm[:, None] * $(cm_stride) +
    offset_cn[None, :] * $(cn_stride)
  c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < $(N))
  tl.store(c_ptr, accumulator.to(tl.float16).to(OUT_DTYPE), mask=c_mask)
}

def bfloatTyped (out_dtype : FloatDType)
    (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  tl.region input_ptr = tl.bfloat16, lora_ptr = tl.bfloat16, out_ptr = OUT_DTYPE
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  cur_batch = tl.program_id(2)
  M = tl.load(seq_lens + cur_batch)
  cur_seq_start = tl.load(b_seq_start_loc + cur_batch)
  lora_index = tl.load(lora_indices + cur_batch)
  offset_m = tl.arange(0, $(BLOCK_M)) + pid_m * $(BLOCK_M)
  offset_n = tl.arange(0, $(BLOCK_N)) + pid_n * $(BLOCK_N)
  offset_k = tl.arange(0, $(BLOCK_K))
  ram = offset_m % M
  rbn = offset_n % $(N)
  a_ptr = input_ptr + cur_seq_start * $(xm_stride) +
    ram[:, None] * $(xm_stride) + offset_k[None, :] * $(xk_stride)
  b_ptr = lora_ptr + $(l0_stride) * lora_index +
    offset_k[:, None] * $(lora_n_stride) + rbn[None, :] * $(lora_k_stride)
  accumulator = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  for kk in range($(0), $(K), $(BLOCK_K)) {
    tiled_a = tl.load(a_ptr)
    tiled_b = tl.load(b_ptr)
    accumulator += tl.dot(tiled_a, tiled_b)
    a_ptr += $(BLOCK_K) * $(xk_stride)
    b_ptr += $(BLOCK_K) * $(lora_n_stride)
  }
  offset_cm = cur_seq_start + offset_m
  offset_cn = offset_n + $(slice_offset)
  c_ptr = out_ptr + offset_cm[:, None] * $(cm_stride) +
    offset_cn[None, :] * $(cn_stride)
  c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < $(N))
  tl.store(c_ptr, accumulator.to(tl.bfloat16).to(OUT_DTYPE), mask=c_mask)
}

def source (fmt : Storage) (out : Output) (p : Parameters) : ComputeKernel :=
  let f := match fmt with | .half => halfTyped | .bfloat => bfloatTyped
  f (outputType out) p.input p.weight p.output p.starts p.lengths p.adapters p.columns p.inner
    p.inputRowStride p.inputInnerStride p.adapterStride p.weightColumnStride p.weightInnerStride
    p.outputRowStride p.outputColumnStride p.sliceOffset p.blockRows p.blockColumns p.blockInner

def prelude (fmt : Storage) (out : Output) (p : Parameters) := (source fmt out p).surfaceBody.take 14
def body (fmt : Storage) (out : Output) (p : Parameters) : List ComputeStmt :=
  let code : List ComputeStmt := (source fmt out p).surfaceBody
  match code[14]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []
def suffix (fmt : Storage) (out : Output) (p : Parameters) := (source fmt out p).surfaceBody.drop 15

theorem source_split (fmt : Storage) (out : Output) (p : Parameters) :
    (source fmt out p).surfaceBody = prelude fmt out p ++
      [.forRange "kk" 0 p.inner p.blockInner (body fmt out p)] ++ suffix fmt out p := by
  cases fmt <;> rfl
#axiomsClean source_split
end
end VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
