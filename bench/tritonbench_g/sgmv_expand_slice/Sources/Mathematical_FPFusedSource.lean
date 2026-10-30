import bench.tritonbench_g.sgmv_expand_slice.Sources.Mathematical_FPContract
namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
/-! Separate explicit-accumulator graph observed in the compiled one-comma
candidate. No equivalence to the original independent dot+add AST is asserted. -/
def fusedHalfTyped (out_dtype : FloatDType)
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
    accumulator = tl.dot(tiled_a, tiled_b, accumulator)
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

def fusedBfloatTyped (out_dtype : FloatDType)
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
    accumulator = tl.dot(tiled_a, tiled_b, accumulator)
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

def fusedSource (fmt : Storage) (out : Output) (p : Parameters) : ComputeKernel :=
  let f := match fmt with | .half => fusedHalfTyped | .bfloat => fusedBfloatTyped
  f (outputType out) p.input p.weight p.output p.starts p.lengths p.adapters p.columns p.inner
    p.inputRowStride p.inputInnerStride p.adapterStride p.weightColumnStride p.weightInnerStride
    p.outputRowStride p.outputColumnStride p.sliceOffset p.blockRows p.blockColumns p.blockInner

def fusedBody (fmt : Storage) (out : Output) (p : Parameters) : List ComputeStmt :=
  let code : List ComputeStmt := (fusedSource fmt out p).surfaceBody
  match code[14]? with | some (ComputeStmt.forRange _ _ _ _ code) => code | _ => []

theorem fused_split (fmt : Storage) (out : Output) (p : Parameters) :
    (fusedSource fmt out p).surfaceBody = prelude fmt out p ++
      [.forRange "kk" 0 p.inner p.blockInner (fusedBody fmt out p)] ++ suffix fmt out p := by
  cases fmt <;> rfl

/-- The parsed compiled-graph profile contains an actual dotAcc primitive;
its output .real tag is a promoted register under explicit fp32 precision. -/
theorem fused_dot_node (fmt : Storage) (out : Output) (p : Parameters) :
    (fusedBody fmt out p)[2]? = some (.assign .real [p.blockRows,p.blockColumns] "accumulator"
      (.compute (.alg .fp32 (.dotAcc (storageType fmt) (storageType fmt) .real (batch := [])
        (.ref (storageType fmt).toTileDType [p.blockRows,p.blockInner] "tiled_a")
        (.ref (storageType fmt).toTileDType [p.blockInner,p.blockColumns] "tiled_b")
        (.ref .real [p.blockRows,p.blockColumns] "accumulator") {})))) := by
  cases fmt <;> rfl
#axiomsClean fused_dot_node

structure DotAcc where
  apply : Option ComputeDType → (left right output : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → Word) → (TileIndex (batch++[k,n]) → Word) →
    (TileIndex (batch++[m,n]) → Word) → (TileIndex (batch++[m,n]) → Word)

def installAcc (rest : Algebra Word) (F : DotAcc) : Algebra Word :=
  {rest with
    dotAcc := fun p l r o opts {_} {_} {_} {_} a b c =>
      some (fun i => ofFloat o (F.apply p l r o opts (fun j => toFloat l (a j))
        (fun j => toFloat r (b j)) (fun j => toFloat o (c j)) i))}

def unusedDot : Dot := ⟨fun _ _ _ _ _ {_} {_} {_} {_} _ _ _ => FP.Value.qNaN .fp32⟩
def fusedBackend (rest : Algebra Word) (cfg : FP.Config) (F : DotAcc) := backend (installAcc rest F) cfg unusedDot

def fusedUpdate [Inhabited Word] (F : DotAcc) (fmt : Storage) (p : Parameters) (s : State Word)
    (cur : Nat) (acc : Accumulator p) : Accumulator p :=
  F.apply (some .fp32) (storageType fmt) (storageType fmt) .real {} (batch := [])
    (aValues fmt p s cur) (bValues fmt p s cur) acc

def fusedAccumulated [Inhabited Word] (F : DotAcc) (fmt : Storage) (p : Parameters) (s : State Word) :=
  ParallelAttention.FPStages.foldRange (fusedUpdate F fmt p s) 0 p.inner p.blockInner (initial p)
#axiomsClean fused_split
end
end VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
