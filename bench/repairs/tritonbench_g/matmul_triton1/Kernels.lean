import bench.repairs.tritonbench_g.matmul_triton1.revision001.Kernels

/-! Shared readable selected-source entry. The uint64 ABI/index execution is
proved in revision001.FiniteSource; this is the natural typed quotation of the
same masked 32x32x32 program and selected live FP32 FMA carry. -/
namespace VeriTile.Bench.Repairs.MatmulTriton1.Kernel
open VeriTile.Triton
def source (p : Revision001.Core.Parameters) : ComputeKernel :=
  let input := p.input
  let weight := p.weight
  let output := p.output
  triton {
  tl.region input = tl.float32, weight = tl.float32, output = tl.float32
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(p.columns), $(p.blockColumns))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = pid_m * $(p.blockRows) + tl.arange(0, $(p.blockRows))
  offs_bn = pid_n * $(p.blockColumns) + tl.arange(0, $(p.blockColumns))
  offs_k = tl.arange(0, $(p.blockInner))
  a_ptrs = input + offs_am[:, None] * $(p.inputRow) + offs_k[None, :] * $(p.inputInner)
  b_ptrs = weight + offs_k[:, None] * $(p.weightInner) + offs_bn[None, :] * $(p.weightColumn)
  accumulator = tl.zeros([$(p.blockRows), $(p.blockColumns)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(p.innerSize), $(p.blockInner)), $(1)) {
    a = tl.load(a_ptrs, mask=(offs_am[:, None] < $(p.rows)) & (offs_k[None, :] < $(p.innerSize) - k * $(p.blockInner)), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < $(p.innerSize) - k * $(p.blockInner)) & (offs_bn[None, :] < $(p.columns)), other=0.0)
    accumulator = tl.dot(a, b, accumulator, allow_tf32=false)
    a_ptrs += $(p.blockInner) * $(p.inputInner)
    b_ptrs += $(p.blockInner) * $(p.weightInner)
  }
  c_ptrs = output + offs_am[:, None] * $(p.outputRow) + offs_bn[None, :] * $(p.outputColumn)
  c_mask = (offs_am[:, None] < $(p.rows)) & (offs_bn[None, :] < $(p.columns))
  tl.store(c_ptrs, accumulator, mask=c_mask)
}
/-- Constant-first row-block index canonicalization; complete source retained. -/
def rowCanonical (p : Revision001.Core.Parameters) : ComputeKernel :=
  let input := p.input
  let weight := p.weight
  let output := p.output
  triton {
  tl.region input = tl.float32, weight = tl.float32, output = tl.float32
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(p.columns), $(p.blockColumns))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = $(p.blockRows) * pid_m + tl.arange(0, $(p.blockRows))
  offs_bn = pid_n * $(p.blockColumns) + tl.arange(0, $(p.blockColumns))
  offs_k = tl.arange(0, $(p.blockInner))
  a_ptrs = input + offs_am[:, None] * $(p.inputRow) + offs_k[None, :] * $(p.inputInner)
  b_ptrs = weight + offs_k[:, None] * $(p.weightInner) + offs_bn[None, :] * $(p.weightColumn)
  accumulator = tl.zeros([$(p.blockRows), $(p.blockColumns)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(p.innerSize), $(p.blockInner)), $(1)) {
    a = tl.load(a_ptrs, mask=(offs_am[:, None] < $(p.rows)) & (offs_k[None, :] < $(p.innerSize) - k * $(p.blockInner)), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < $(p.innerSize) - k * $(p.blockInner)) & (offs_bn[None, :] < $(p.columns)), other=0.0)
    accumulator = tl.dot(a, b, accumulator, allow_tf32=false)
    a_ptrs += $(p.blockInner) * $(p.inputInner)
    b_ptrs += $(p.blockInner) * $(p.weightInner)
  }
  c_ptrs = output + offs_am[:, None] * $(p.outputRow) + offs_bn[None, :] * $(p.outputColumn)
  c_mask = (offs_am[:, None] < $(p.rows)) & (offs_bn[None, :] < $(p.columns))
  tl.store(c_ptrs, accumulator, mask=c_mask)
}
/-- Cache loop-invariant row/column comparisons for all K loads and the final store. -/
def maskHoisted (p : Revision001.Core.Parameters) : ComputeKernel :=
  let input := p.input
  let weight := p.weight
  let output := p.output
  triton {
  tl.region input = tl.float32, weight = tl.float32, output = tl.float32
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(p.columns), $(p.blockColumns))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
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
    accumulator = tl.dot(a, b, accumulator, allow_tf32=false)
    a_ptrs += $(p.blockInner) * $(p.inputInner)
    b_ptrs += $(p.blockInner) * $(p.weightInner)
  }
  c_ptrs = output + offs_am[:, None] * $(p.outputRow) + offs_bn[None, :] * $(p.outputColumn)
  c_mask = row_mask[:, None] & column_mask[None, :]
  tl.store(c_ptrs, accumulator, mask=c_mask)
}
end VeriTile.Bench.Repairs.MatmulTriton1.Kernel
