import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import bench.tritonbench_g.lora_expand_gemv.Proofs.FPTypedSource
namespace VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
abbrev Format := LoraExpandGemv.FPValue.OutputFormat
abbrev dtype := LoraExpandGemv.FPValue.outputType
inductive Graph where
  | source | accumulator
  deriving DecidableEq, Repr
structure Parameters where
  (output saved input weight bias : RegionName)
  (rows columns outputRowStride outputColStride savedRowStride savedColStride
    inputRowStride inputInnerStride weightColStride weightInnerStride
    blockRows groupRows blockColumns blockInner blocks : Nat)
  (hasBias saveInputs : Bool)
  activation : String
def typed (dtype out_dtype : FloatDType)
    (C ACT_INPUTS A B biasR : RegionName)
    (M N output_m_stride output_n_stride act_inputs_m_stride act_inputs_n_stride
      a_m_stride a_k_stride b_n_stride b_k_stride
      BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String) :
    ComputeKernel := triton {
  tl.region C = DTYPE, ACT_INPUTS = OUT_DTYPE, A = DTYPE, B = DTYPE, biasR = DTYPE
  program_idx = tl.program_id(axis=0)
  grid_m = ($((M : Nat)) + $(BLOCK_M) - $(1)) // $(BLOCK_M)
  grid_n = ($((N : Nat)) + $(BLOCK_N) - $(1)) // $(BLOCK_N)
  width = $(GROUP_M) * grid_n
  group_idx = program_idx // width
  group_size = min(grid_m - group_idx * $(GROUP_M), $(GROUP_M))
  block_m_idx = group_idx * $(GROUP_M) + (program_idx % group_size)
  block_n_idx = (program_idx % width) // group_size
  m_offs_untagged = block_m_idx * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offs_untagged = block_n_idx * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  m_offs = tl.max_contiguous(tl.multiple_of(m_offs_untagged % $(M), $(BLOCK_M)), $(BLOCK_M))
  n_offs = tl.max_contiguous(tl.multiple_of(n_offs_untagged % $(N), $(BLOCK_N)), $(BLOCK_N))
  k_range_offs = tl.arange(0, $(BLOCK_K))
  A = A + (m_offs[:, None] * $(a_m_stride) + k_range_offs[None, :] * $(a_k_stride))
  B = B + (k_range_offs[:, None] * $(b_k_stride) + n_offs[None, :] * $(b_n_stride))
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  if HAS_BIAS {
    bias = tl.load(biasR + n_offs, mask=n_offs < $(N), other=0.0).to(tl.float32)
    acc += bias[None, :]
  }
  for k in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(A)
    b = tl.load(B)
    acc += tl.dot(a, b)
    A += $(BLOCK_K) * $(a_k_stride)
    B += $(BLOCK_K) * $(b_k_stride)
  }
  if SHOULD_SAVE_ACT_INPUTS {
    act_in_ptrs = ACT_INPUTS + m_offs[:, None] * $(act_inputs_m_stride) + n_offs[None, :] * $(act_inputs_n_stride)
    tl.store(act_in_ptrs, acc.to(OUT_DTYPE))
  }
  if ACTIVATION == "tanh" {
    acc = tanh(acc)
  }
  if ACTIVATION == "gelu" {
    acc = acc * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(acc / $((Real.sqrt 2 : ℝ))))
  }
  if ACTIVATION == "fast_gelu" {
    acc = 0.5 * acc * (1 + tanh($((Real.sqrt (2.0 / Real.pi) : ℝ)) * (acc + 0.044715 * acc * acc * acc)))
  }
  if ACTIVATION == "relu" {
    acc = tl.maximum(0, acc)
  }
  C = C + m_offs[:, None] * $(output_m_stride) + n_offs[None, :] * $(output_n_stride)
  c_ptr_mask = (m_offs < $(M))[:, None] & (n_offs < $(N))[None, :]
  tl.store(C, acc.to(DTYPE), mask=c_ptr_mask)
}

def fusedTyped (dtype out_dtype : FloatDType)
    (C ACT_INPUTS A B biasR : RegionName)
    (M N output_m_stride output_n_stride act_inputs_m_stride act_inputs_n_stride
      a_m_stride a_k_stride b_n_stride b_k_stride
      BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String) :
    ComputeKernel := triton {
  tl.region C = DTYPE, ACT_INPUTS = OUT_DTYPE, A = DTYPE, B = DTYPE, biasR = DTYPE
  program_idx = tl.program_id(axis=0)
  grid_m = ($((M : Nat)) + $(BLOCK_M) - $(1)) // $(BLOCK_M)
  grid_n = ($((N : Nat)) + $(BLOCK_N) - $(1)) // $(BLOCK_N)
  width = $(GROUP_M) * grid_n
  group_idx = program_idx // width
  group_size = min(grid_m - group_idx * $(GROUP_M), $(GROUP_M))
  block_m_idx = group_idx * $(GROUP_M) + (program_idx % group_size)
  block_n_idx = (program_idx % width) // group_size
  m_offs_untagged = block_m_idx * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offs_untagged = block_n_idx * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  m_offs = tl.max_contiguous(tl.multiple_of(m_offs_untagged % $(M), $(BLOCK_M)), $(BLOCK_M))
  n_offs = tl.max_contiguous(tl.multiple_of(n_offs_untagged % $(N), $(BLOCK_N)), $(BLOCK_N))
  k_range_offs = tl.arange(0, $(BLOCK_K))
  A = A + (m_offs[:, None] * $(a_m_stride) + k_range_offs[None, :] * $(a_k_stride))
  B = B + (k_range_offs[:, None] * $(b_k_stride) + n_offs[None, :] * $(b_n_stride))
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  if HAS_BIAS {
    bias = tl.load(biasR + n_offs, mask=n_offs < $(N), other=0.0).to(tl.float32)
    acc = bias[None, :] + acc
  }
  for k in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(A)
    b = tl.load(B)
    acc = tl.dot(a, b, acc)
    A += $(BLOCK_K) * $(a_k_stride)
    B += $(BLOCK_K) * $(b_k_stride)
  }
  if SHOULD_SAVE_ACT_INPUTS {
    act_in_ptrs = ACT_INPUTS + m_offs[:, None] * $(act_inputs_m_stride) + n_offs[None, :] * $(act_inputs_n_stride)
    tl.store(act_in_ptrs, acc.to(OUT_DTYPE))
  }
  if ACTIVATION == "tanh" {
    acc = tanh(acc)
  }
  if ACTIVATION == "gelu" {
    acc = acc * 0.5 * (tl.extra.cuda.libdevice.erf(acc / $((Real.sqrt 2 : ℝ))) + 1.0)
  }
  if ACTIVATION == "fast_gelu" {
    acc = acc * 0.5 * (tanh((acc + acc * 0.044715 * acc * acc) * $((Real.sqrt (2.0 / Real.pi) : ℝ))) + 1)
  }
  if ACTIVATION == "relu" {
    acc = tl.maximum(acc, 0)
  }
  C = C + m_offs[:, None] * $(output_m_stride) + n_offs[None, :] * $(output_n_stride)
  c_ptr_mask = (m_offs < $(M))[:, None] & (n_offs < $(N))[None, :]
  tl.store(C, acc.to(DTYPE), mask=c_ptr_mask)
}

def source (g : Graph) (fmt saved : Format) (p : Parameters) : ComputeKernel :=
  let f := match g with | .source => typed | .accumulator => fusedTyped
  f (dtype fmt) (dtype saved) p.output p.saved p.input p.weight p.bias p.rows p.columns
    p.outputRowStride p.outputColStride p.savedRowStride p.savedColStride
    p.inputRowStride p.inputInnerStride p.weightColStride p.weightInnerStride
    p.blockRows p.groupRows p.blockColumns p.blockInner p.blocks p.hasBias p.saveInputs p.activation

def scalars (p : Parameters) : List ComputeStmt := (source .source .half .half p).surfaceBody.take 13
def seeds (g : Graph) (fmt saved : Format) (p : Parameters) : List ComputeStmt :=
  ((source g fmt saved p).surfaceBody.drop 13).take 4
def body (g : Graph) (fmt saved : Format) (p : Parameters) : List ComputeStmt :=
  let code : List ComputeStmt := (source g fmt saved p).surfaceBody
  match code[17]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code | _ => []
def suffix (g : Graph) (fmt saved : Format) (p : Parameters) : List ComputeStmt :=
  (source g fmt saved p).surfaceBody.drop 18

theorem source_split (g : Graph) (fmt saved : Format) (p : Parameters) :
    (source g fmt saved p).surfaceBody = scalars p ++ seeds g fmt saved p ++
      [.forRange "k" 0 p.blocks 1 (body g fmt saved p)] ++ suffix g fmt saved p := by
  cases g <;> rfl
#axiomsClean source_split
end
end VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
