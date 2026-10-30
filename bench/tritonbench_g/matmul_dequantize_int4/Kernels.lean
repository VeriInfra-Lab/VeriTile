import bench.tritonbench_g.matmul_dequantize_int4.Sources.Mathematical_FPReference
import bench.tritonbench_g.matmul_kernel.Kernels
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4
abbrev View := VeriTile.Bench.Repairs.MatmulKernel.View
inductive Opcode where
  | sync16x8 | wg64x32 | wg64x64 | wg64x128 | wg64x256
  deriving DecidableEq, Repr
structure Tuning where
  BM : Nat
  BN : Nat
  BK : Nat
  warps : Nat
  stages : Nat
  opcode : Opcode
  deriving DecidableEq, Repr
/-- Every literal public autotune entry, in source order. GROUP_SIZE_M=8
for every entry. Default fusion is enabled; no configuration is removed. -/
def tunings : Fin 17 → Tuning := ![
  ⟨64,256,32,4,4,.wg64x256⟩, ⟨128,128,32,4,4,.wg64x128⟩,
  ⟨128,64,32,4,4,.wg64x64⟩, ⟨64,128,32,4,4,.wg64x128⟩,
  ⟨128,32,32,4,4,.wg64x32⟩, ⟨64,64,32,4,4,.wg64x64⟩,
  ⟨64,128,32,8,2,.wg64x64⟩, ⟨64,64,64,8,3,.wg64x32⟩,
  ⟨64,256,64,4,4,.wg64x256⟩, ⟨128,128,64,4,4,.wg64x128⟩,
  ⟨128,64,64,4,4,.wg64x64⟩, ⟨64,128,64,4,4,.wg64x128⟩,
  ⟨128,32,64,4,4,.wg64x32⟩, ⟨64,64,64,4,4,.wg64x64⟩,
  ⟨64,128,64,8,2,.wg64x64⟩, ⟨32,32,128,4,2,.sync16x8⟩,
  ⟨64,64,128,8,3,.wg64x32⟩]

/-- Selected documented public contract: A/scales/output half; packed inputs
signed int32; contiguous two-dimensional A; separated positive-stride views
for Q, scales, zeros and output. Different named regions do not alias. -/
structure Call where
  M : Nat
  N : Nat
  K : Nat
  group : Nat
  A : View
  Q : View
  S : View
  Z : View
  C : View
  supplied : Bool
structure Call.Valid (c : Call) : Prop where
  m : 0<c.M
  n : 0<c.N
  k : 0<c.K
  g : 0<c.group
  n256 : 256∣c.N
  k128 : 128∣c.K
  g128 : 128∣c.group
  groups : c.group∣c.K
  aShape : c.A.rows=c.M ∧ c.A.cols=c.K
  qShape : c.Q.rows=c.K/8 ∧ c.Q.cols=c.N
  sShape : c.S.rows=c.K/c.group ∧ c.S.cols=c.N
  zShape : c.Z.rows=c.K/c.group ∧ c.Z.cols=c.N/8
  cShape : c.C.rows=c.M ∧ c.C.cols=c.N
  aContiguous : c.A.sr=c.K ∧ c.A.sc=1
  aValid : c.A.Valid
  qValid : c.Q.Valid
  sValid : c.S.Valid
  zValid : c.Z.Valid
  cValid : c.C.Valid
  freshContiguous : c.supplied=Bool.false → c.C.sr=c.N ∧ c.C.sc=1

def params (c : Call) (i : Fin 17) : FPTyped.Parameters where
  input := "A"
  output := "C"
  scale := "S"
  weight := "Q"
  zero := "Z"
  rows := c.M
  columns := c.N
  inner := c.K
  inputRowStride := c.A.sr
  inputInnerStride := c.A.sc
  weightInnerStride := c.Q.sr
  weightColumnStride := c.Q.sc
  outputRowStride := c.C.sr
  outputColumnStride := c.C.sc
  scaleGroupStride := c.S.sr
  scaleColumnStride := c.S.sc
  zeroGroupStride := c.Z.sr
  zeroColumnStride := c.Z.sc
  groupSize := c.group
  blockRows := (tunings i).BM
  blockColumns := (tunings i).BN
  blockInner := (tunings i).BK
  groupRows := 8
  noGroups := decide (c.group=c.K)

noncomputable section


/-- Complete candidate quotation; only the group extent is precomputed. -/
def groupedTyped (dtype : FloatDType)
    (a_ptr c_ptr scales_ptr : RegionName) (b_ptr zeros_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize : Nat) (NO_GROUPS : Bool)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ComputeKernel := triton {
  tl.region a_ptr = DTYPE, c_ptr = DTYPE, scales_ptr = DTYPE, b_ptr = tl.int32, zeros_ptr = tl.int32

  bits = $(4)
  infearure_per_bits = $(8)
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_k = tl.cdiv($(K), $(BLOCK_SIZE_K))
  num_pid_in_group = $(GROUP_SIZE_M * ((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N))
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_bn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = a_ptr + (offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak))
  a_mask = (offs_am[:, None] < $(M))
  b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * $(stride_bk) + offs_bn[None, :] * $(stride_bn))
  scales_ptrs = scales_ptr + offs_bn * $(stride_scales_n)
  zeros_ptrs = zeros_ptr + ((offs_bn // infearure_per_bits) * $(stride_zeros_n))
  shifter = (offs_k % infearure_per_bits) * bits
  zeros_shifter = (offs_bn % infearure_per_bits) * bits
  if NO_GROUPS {
    scales = tl.load(scales_ptrs)
    zeros = tl.load(zeros_ptrs)
    zeros = (zeros >> zeros_shifter) & 15
    zeros = tl.toReal(zeros).to(DTYPE) * scales
  }
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), num_pid_k, $(1)) {
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs)
    if not NO_GROUPS {
      g_id = k // ($(groupsize) // $(BLOCK_SIZE_K))
      ptr = scales_ptrs + g_id * $(stride_scales_g)
      scales = tl.load(ptr)
      ptr = zeros_ptrs + g_id * $(stride_zeros_g)
      zeros = tl.load(ptr)
      zeros = (zeros >> zeros_shifter) & 15
      zeros = tl.toReal(zeros).to(DTYPE) * scales
    }
    b = (b >> shifter[:, None]) & 15
    b = tl.toReal(b).to(DTYPE) * scales[None, :] - zeros[None, :]
    accumulator += tl.dot(a, (b).to(a.dtype))
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += ($(BLOCK_SIZE_K) // infearure_per_bits) * $(stride_bk)
  }
  c = (accumulator).to(c_ptr.dtype.element_ty)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = c_ptr + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, accumulator.to(DTYPE), mask=c_mask)
}


/-- Complete candidate quotation; only the group extent is precomputed. -/
def groupedCompiledTyped (dtype : FloatDType)
    (a_ptr c_ptr scales_ptr : RegionName) (b_ptr zeros_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize : Nat) (NO_GROUPS : Bool)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ComputeKernel := triton {
  tl.region a_ptr = DTYPE, c_ptr = DTYPE, scales_ptr = DTYPE, b_ptr = tl.int32, zeros_ptr = tl.int32

  bits = $(4)
  infearure_per_bits = $(8)
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_k = tl.cdiv($(K), $(BLOCK_SIZE_K))
  num_pid_in_group = $(GROUP_SIZE_M * ((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N))
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_bn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = a_ptr + (offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak))
  a_mask = (offs_am[:, None] < $(M))
  b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * $(stride_bk) + offs_bn[None, :] * $(stride_bn))
  scales_ptrs = scales_ptr + offs_bn * $(stride_scales_n)
  zeros_ptrs = zeros_ptr + ((offs_bn // infearure_per_bits) * $(stride_zeros_n))
  shifter = (offs_k % infearure_per_bits) * bits
  zeros_shifter = (offs_bn % infearure_per_bits) * bits
  if NO_GROUPS {
    scales = tl.load(scales_ptrs)
    zeros = tl.load(zeros_ptrs)
    zeros = (zeros >> zeros_shifter) & 15
    zeros = tl.toReal(zeros).to(DTYPE) * scales
  }
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), num_pid_k, $(1)) {
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs)
    if not NO_GROUPS {
      g_id = k // ($(groupsize) // $(BLOCK_SIZE_K))
      ptr = scales_ptrs + g_id * $(stride_scales_g)
      scales = tl.load(ptr)
      ptr = zeros_ptrs + g_id * $(stride_zeros_g)
      zeros = tl.load(ptr)
      zeros = (zeros >> zeros_shifter) & 15
      zeros = tl.toReal(zeros).to(DTYPE) * scales
    }
    b = (b >> shifter[:, None]) & 15
    b = tl.toReal(b).to(DTYPE) * scales[None, :] - zeros[None, :]
    accumulator = tl.dot(a, (b).to(a.dtype), accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += ($(BLOCK_SIZE_K) // infearure_per_bits) * $(stride_bk)
  }
  c = (accumulator).to(c_ptr.dtype.element_ty)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = c_ptr + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, accumulator.to(DTYPE), mask=c_mask)
}


def groupedSource (p : FPTyped.Parameters) : ComputeKernel := groupedTyped .fp16
  p.input p.output p.scale p.weight p.zero p.rows p.columns p.inner
  p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.scaleGroupStride p.scaleColumnStride
  p.zeroGroupStride p.zeroColumnStride p.groupSize p.noGroups
  p.blockRows p.blockColumns p.blockInner p.groupRows

def groupedCompiledSource (p : FPTyped.Parameters) : ComputeKernel := groupedCompiledTyped .fp16
  p.input p.output p.scale p.weight p.zero p.rows p.columns p.inner
  p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.scaleGroupStride p.scaleColumnStride
  p.zeroGroupStride p.zeroColumnStride p.groupSize p.noGroups
  p.blockRows p.blockColumns p.blockInner p.groupRows

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4
