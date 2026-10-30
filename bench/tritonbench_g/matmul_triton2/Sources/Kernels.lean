import VeriTile.Triton
import bench.tritonbench_g.matmul_triton1.Sources.Kernels

namespace VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Core
open VeriTile.Triton
abbrev Parameters := VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core.Parameters
def inner (p : Parameters) : Nat := p.innerSize
def steps (p : Parameters) : Nat := (inner p+p.blockInner-1)/p.blockInner

/-- Repaired masks and all original grouped-grid calculations. This is the
compiler-selected dot-accumulator graph: offline TTIR binds the original
`accumulator += tl.dot(a,b)` to an accumulator-fed dot with tf32 inputs.
No source precision/fusion override is inserted. -/
def typed (p : Parameters) : ComputeKernel :=
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
  a_ptrs = input + offs_am[:, None] * $(p.inputRow) + offs_k[None, :] * $(p.inputInner)
  b_ptrs = weight + offs_k[:, None] * $(p.weightInner) + offs_bn[None, :] * $(p.weightColumn)
  accumulator = tl.zeros([$(p.blockRows), $(p.blockColumns)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(inner p), $(p.blockInner)), $(1)) {
    a = tl.load(a_ptrs, mask=(offs_am[:, None] < $(p.rows)) & (offs_k[None, :] < $(inner p) - k * $(p.blockInner)), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < $(inner p) - k * $(p.blockInner)) & (offs_bn[None, :] < $(p.columns)), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(p.blockInner) * $(p.inputInner)
    b_ptrs += $(p.blockInner) * $(p.weightInner)
  }
  c_ptrs = output + offs_am[:, None] * $(p.outputRow) + offs_bn[None, :] * $(p.outputColumn)
  c_mask = (offs_am[:, None] < $(p.rows)) & (offs_bn[None, :] < $(p.columns))
  tl.store(c_ptrs, accumulator, mask=c_mask)
}
def source (p : Parameters) : ComputeKernel := typed p
end VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Core

namespace VeriTile.Bench.Repairs.MatmulTriton2.Revision001
open VeriTile.Triton
def matmul_triton2_kernel (p : Core.Parameters) : ComputeKernel := Core.source p
end VeriTile.Bench.Repairs.MatmulTriton2.Revision001
