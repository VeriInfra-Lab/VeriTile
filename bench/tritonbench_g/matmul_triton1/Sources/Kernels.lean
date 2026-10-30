import VeriTile.Triton

namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
open VeriTile.Triton
structure Parameters where
  (input weight output : RegionName)
  (rows columns innerSize inputRow inputInner weightInner weightColumn outputRow outputColumn
    blockRows blockColumns blockInner groupRows : Nat)
def inner (p : Parameters) : Nat := p.innerSize
def steps (p : Parameters) : Nat := (inner p+p.blockInner-1)/p.blockInner

/-- Repaired masks and the original row-major program ordering. This is the
compiler-selected dot-accumulator graph: offline TTIR binds the original
`z += tl.dot(x_sub,y_sub,allow_tf32=False)` to an accumulator-fed IEEE dot.
No source precision/fusion override is inserted. -/
def typed (p : Parameters) : ComputeKernel :=
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
  for k in range($(0), tl.cdiv($(inner p), $(p.blockInner)), $(1)) {
    a = tl.load(a_ptrs, mask=(offs_am[:, None] < $(p.rows)) & (offs_k[None, :] < $(inner p) - k * $(p.blockInner)), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < $(inner p) - k * $(p.blockInner)) & (offs_bn[None, :] < $(p.columns)), other=0.0)
    accumulator = tl.dot(a, b, accumulator, allow_tf32=false)
    a_ptrs += $(p.blockInner) * $(p.inputInner)
    b_ptrs += $(p.blockInner) * $(p.weightInner)
  }
  c_ptrs = output + offs_am[:, None] * $(p.outputRow) + offs_bn[None, :] * $(p.outputColumn)
  c_mask = (offs_am[:, None] < $(p.rows)) & (offs_bn[None, :] < $(p.columns))
  tl.store(c_ptrs, accumulator, mask=c_mask)
}
def source (p : Parameters) : ComputeKernel := typed p
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core

namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001
open VeriTile.Triton
def matmul_triton1_kernel (p : Core.Parameters) : ComputeKernel := Core.source p
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001
