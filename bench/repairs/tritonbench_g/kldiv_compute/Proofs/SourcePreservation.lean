import bench.repairs.tritonbench_g.kldiv_compute.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KldivCompute.KernelDSL.SourcePreservation
open VeriTile.Triton

def previous_kernel (p : Parameters) : ComputeKernel :=
  let X := p.x
  let Y := p.y
  let O := p.output
  triton {
    tl.region X = tl.float32, Y = tl.float32, O = tl.float32
    pid = tl.program_id(0)
    block_start = pid * $(p.block)
    offsets = block_start + tl.arange(0, $(p.block))
    mask = offsets < $(p.n)
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    output = x * tl.log(x / y)
    tl.store(O + offsets, output, mask=mask)
  }

def previous_commutedKernel (p : Parameters) : ComputeKernel :=
  let X := p.x
  let Y := p.y
  let O := p.output
  triton {
    tl.region X = tl.float32, Y = tl.float32, O = tl.float32
    pid = tl.program_id(0)
    block_start = $(p.block) * pid
    offsets = block_start + tl.arange(0, $(p.block))
    mask = offsets < $(p.n)
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    output = x * tl.log(x / y)
    tl.store(O + offsets, output, mask=mask)
  }


theorem kernel_preserved (p : Parameters) : kernel p = previous_kernel p := rfl
theorem commutedKernel_preserved (p : Parameters) : commutedKernel p = previous_commutedKernel p := rfl

#axiomsClean kernel_preserved
#axiomsClean commutedKernel_preserved
end VeriTile.Bench.Repairs.KldivCompute.KernelDSL.SourcePreservation
