import bench.repairs.tritonbench_g.matmul_triton2.revision001.Kernels

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly
/-- The readable shared-DSL quotation is the frozen task-specific source in
revision001/Kernels.lean. It retains grouped coordinates, both input masks,
K padding, accumulator-fed TF32 dot and the output mask. The finite64 adapter
and actual constexpr binder are audited separately. -/
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
