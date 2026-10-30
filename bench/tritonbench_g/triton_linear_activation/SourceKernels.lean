import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import bench.tritonbench_g.triton_linear_activation.Proofs.Config

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelSource
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

/-- Concrete arguments after successful ordinary preparation. The actual
kernel pointer formats are FP32 normally and FP16 under the explicit AMP cast.
The caller's FP32 saved tensor and any AMP copyback are outside this kernel. -/
structure Parameters where
  (output saved input weight bias : RegionName)
  (rows columns inner : Nat)
  (outputRowStride outputColStride savedRowStride savedColStride : Nat)
  (inputRowStride inputInnerStride weightColStride weightInnerStride : Nat)
  (blockRows groupRows blockColumns blockInner : Nat)
  (hasBias saveInputs : Bool)
  activation : Option String

def dtype : Mode → FloatDType | .fp32 => .fp32 | .autocastFP16 => .fp16

/-- Complete selected kernel. The descending remaining-K loop, two-argument
dot followed by outer addition, operand order, independent save mask, and
unwrapped output coordinates are retained. Store casts make the pointer's
implicit storage conversion explicit in the typed DSL. Helpers are inlined. -/
def typed (dtype : FloatDType)
    (C ACT_INPUTS A B biasR : RegionName)
    (M N K output_m_stride output_n_stride act_inputs_m_stride act_inputs_n_stride
      a_m_stride a_k_stride b_n_stride b_k_stride BLOCK_M GROUP_M BLOCK_N BLOCK_K : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS K_LOAD_MASK_NEEDED : Bool) (ACTIVATION : String) :
    ComputeKernel := triton {
  tl.region C = DTYPE, ACT_INPUTS = DTYPE, A = DTYPE, B = DTYPE, biasR = DTYPE
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
  m_offs = m_offs_untagged % $(M)
  n_offs = n_offs_untagged % $(N)
  k_range_offs = tl.arange(0, $(BLOCK_K))
  A = A + (m_offs[:, None] * $(a_m_stride) + k_range_offs[None, :] * $(a_k_stride))
  B = B + (k_range_offs[:, None] * $(b_k_stride) + n_offs[None, :] * $(b_n_stride))
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  if HAS_BIAS {
    bias = tl.load(biasR + n_offs, mask=n_offs < $(N), other=0.0).to(tl.float32)
    acc += bias[None, :]
  }
  for k in range($(K), -$(0), -$(BLOCK_K)) {
    if K_LOAD_MASK_NEEDED {
      a = tl.load(A)
      b = tl.load(B)
    } else {
      a = tl.load(A, mask=k_range_offs[None, :] < k, other=0.0)
      b = tl.load(B, mask=k_range_offs[:, None] < k, other=0.0)
    }
    acc += tl.dot(a, b)
    A += $(BLOCK_K) * $(a_k_stride)
    B += $(BLOCK_K) * $(b_k_stride)
  }
  if SHOULD_SAVE_ACT_INPUTS {
    act_in_ptrs = ACT_INPUTS + m_offs_untagged[:, None] * $(act_inputs_m_stride) + n_offs_untagged[None, :] * $(act_inputs_n_stride)
    save_mask = (m_offs_untagged < $(M))[:, None] & (n_offs_untagged < $(N))[None, :]
    tl.store(act_in_ptrs, acc.to(DTYPE), mask=save_mask)
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
  C = C + m_offs_untagged[:, None] * $(output_m_stride) + n_offs_untagged[None, :] * $(output_n_stride)
  c_ptr_mask = (m_offs_untagged < $(M))[:, None] & (n_offs_untagged < $(N))[None, :]
  tl.store(C, acc.to(DTYPE), mask=c_ptr_mask)
}

def source (mode : Mode) (p : Parameters) : ComputeKernel :=
  typed (dtype mode) p.output p.saved p.input p.weight p.bias
    p.rows p.columns p.inner p.outputRowStride p.outputColStride p.savedRowStride p.savedColStride
    p.inputRowStride p.inputInnerStride p.weightColStride p.weightInnerStride
    p.blockRows p.groupRows p.blockColumns p.blockInner p.hasBias p.saveInputs
    (p.inner%p.blockInner==0) (p.activation.getD "")


end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelSource
