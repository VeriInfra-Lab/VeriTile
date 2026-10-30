import bench.tritonbench_g.matmul_triton2.Sources.Kernels
import VeriTile.Triton.DSL.CallSignature

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly
/-- Selected compiler graph, including grouped coordinates, both input
masks, K padding, accumulator-fed TF32 dot and the output mask. The separate
pythonBody below retains the original source-level dot and following addition.
Their FP connection is not established by these definitions. -/
abbrev kernel := Revision001.Core.source
open VeriTile.Triton
/-- Complete candidate with the static row-block factor first. -/
def rowCanonical (p : Revision001.Core.Parameters) : ComputeKernel :=
  let input := p.input
  let weight := p.weight
  let output := p.output
  triton {
  tl.region input = tl.float32, weight = tl.float32, output = tl.float32
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(p.rows), $(p.blockRows))
  num_pid_n = tl.cdiv($(p.columns), $(p.blockColumns))
  num_pid_in_group = $(p.groupRows) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(p.groupRows)
  group_size_m = min(num_pid_m - first_pid_m, $(p.groupRows))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = $(p.blockRows) * pid_m + tl.arange(0, $(p.blockRows))
  offs_bn = pid_n * $(p.blockColumns) + tl.arange(0, $(p.blockColumns))
  offs_k = tl.arange(0, $(p.blockInner))
  a_ptrs = input + offs_am[:, None] * $(p.inputRow) + offs_k[None, :] * $(p.inputInner)
  b_ptrs = weight + offs_k[:, None] * $(p.weightInner) + offs_bn[None, :] * $(p.weightColumn)
  accumulator = tl.zeros([$(p.blockRows), $(p.blockColumns)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(p.innerSize), $(p.blockInner)), $(1)) {
    a = tl.load(a_ptrs, mask=(offs_am[:, None] < $(p.rows)) & (offs_k[None, :] < $(p.innerSize) - k * $(p.blockInner)), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < $(p.innerSize) - k * $(p.blockInner)) & (offs_bn[None, :] < $(p.columns)), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(p.blockInner) * $(p.inputInner)
    b_ptrs += $(p.blockInner) * $(p.weightInner)
  }
  c_ptrs = output + offs_am[:, None] * $(p.outputRow) + offs_bn[None, :] * $(p.outputColumn)
  c_mask = (offs_am[:, None] < $(p.rows)) & (offs_bn[None, :] < $(p.columns))
  tl.store(c_ptrs, accumulator, mask=c_mask)
}
/-- Reuse invariant row/column masks across every K-loop load and final store. -/
def maskHoisted (p : Revision001.Core.Parameters) : ComputeKernel :=
  let input := p.input
  let weight := p.weight
  let output := p.output
  triton {
  tl.region input = tl.float32, weight = tl.float32, output = tl.float32
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(p.rows), $(p.blockRows))
  num_pid_n = tl.cdiv($(p.columns), $(p.blockColumns))
  num_pid_in_group = $(p.groupRows) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(p.groupRows)
  group_size_m = min(num_pid_m - first_pid_m, $(p.groupRows))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = pid_m * $(p.blockRows) + tl.arange(0, $(p.blockRows))
  offs_bn = pid_n * $(p.blockColumns) + tl.arange(0, $(p.blockColumns))
  offs_k = tl.arange(0, $(p.blockInner))
  row_mask = offs_am < $(p.rows)
  column_mask = offs_bn < $(p.columns)
  a_ptrs = input + offs_am[:, None] * $(p.inputRow) + offs_k[None, :] * $(p.inputInner)
  b_ptrs = weight + offs_k[:, None] * $(p.weightInner) + offs_bn[None, :] * $(p.weightColumn)
  accumulator = tl.zeros([$(p.blockRows), $(p.blockColumns)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(p.innerSize), $(p.blockInner)), $(1)) {
    a = tl.load(a_ptrs, mask=row_mask[:, None] & (offs_k[None, :] < $(p.innerSize) - k * $(p.blockInner)), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < $(p.innerSize) - k * $(p.blockInner)) & column_mask[None, :], other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(p.blockInner) * $(p.inputInner)
    b_ptrs += $(p.blockInner) * $(p.weightInner)
  }
  c_ptrs = output + offs_am[:, None] * $(p.outputRow) + offs_bn[None, :] * $(p.outputColumn)
  c_mask = row_mask[:, None] & column_mask[None, :]
  tl.store(c_ptrs, accumulator, mask=c_mask)
}
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly
open VeriTile.Triton
/-- Source-level interface; runtime integers are explicitly cast to uint64
by the body. This interface does not certify a native launch or compiler pass. -/
def pythonCallSignature : CallSignature where
  pointers := [("a_ptr",.fp32),("b_ptr",.fp32),("c_ptr",.fp32)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),("stride_am",.integer),("stride_ak",.integer),("stride_bk",.integer),("stride_bn",.integer),("stride_cm",.integer),("stride_cn",.integer)]
  constexpr := ["BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K","GROUP_SIZE_M"]

/-- Complete Python-source arithmetic and uint64 casts. Its standalone dot
and following addition are distinct from the selected compiler graph above.
The existing Correct/FPEquiv entries continue to concern that selected graph. -/
def pythonBody (a_ptr b_ptr c_ptr : RegionName) (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) : ComputeKernel :=
  with_triton_signature pythonCallSignature in
  triton {
    M = tl.cast(M, tl.uint64)
    N = tl.cast(N, tl.uint64)
    K = tl.cast(K, tl.uint64)
    stride_am = tl.cast(stride_am, tl.uint64)
    stride_ak = tl.cast(stride_ak, tl.uint64)
    stride_bk = tl.cast(stride_bk, tl.uint64)
    stride_bn = tl.cast(stride_bn, tl.uint64)
    stride_cm = tl.cast(stride_cm, tl.uint64)
    stride_cn = tl.cast(stride_cn, tl.uint64)
    pid = tl.program_id(axis=0).to(tl.uint64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.cast(min(num_pid_m - first_pid_m, GROUP_SIZE_M), tl.uint64)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.uint64)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.uint64)
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.uint64)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range($(0), tl.cdiv(K, BLOCK_SIZE_K)) {
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    }

    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
  }

/-- Call-site values are bound separately, before executing the source body. -/
def pythonArguments (p : Revision001.Core.Parameters) : List ScalarArgument :=
  [ScalarArgument.integer "M" (Int.ofNat p.rows),
   ScalarArgument.integer "N" (Int.ofNat p.columns),
   ScalarArgument.integer "K" (Int.ofNat p.innerSize),
   ScalarArgument.integer "stride_am" (Int.ofNat p.inputRow),
   ScalarArgument.integer "stride_ak" (Int.ofNat p.inputInner),
   ScalarArgument.integer "stride_bk" (Int.ofNat p.weightInner),
   ScalarArgument.integer "stride_bn" (Int.ofNat p.weightColumn),
   ScalarArgument.integer "stride_cm" (Int.ofNat p.outputRow),
   ScalarArgument.integer "stride_cn" (Int.ofNat p.outputColumn)]

def pythonSource (p : Revision001.Core.Parameters) : ComputeKernel :=
  (pythonBody p.input p.weight p.output p.blockRows p.blockColumns p.blockInner p.groupRows).withScalarArguments (pythonArguments p)
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly
