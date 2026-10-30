import bench.repairs.tritonbench_g.kldiv_compute.Proofs.HistoricalKernels

namespace VeriTile.Bench.Repairs.KldivCompute.KernelDSL
open VeriTile.Triton

/-- Complete selected 1024-lane source. The unchanged source uses signed32
PID/products/offsets/mask, proved exact under its existing numel < 2^31 guard.
The omitted load padding remains arbitrary and cannot reach an active store.
FP32 division, tl.log, then FP32 multiplication retain their source order. -/
def kernel (p : Parameters) : ComputeKernel :=
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

/-- Existing address-commutation candidate from the original task. Only the
integer block-start operands are swapped; the FP32 computation is unchanged. -/
def commutedKernel (p : Parameters) : ComputeKernel :=
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

/-- Literal typed AST equality permits reuse of the complete source proof. -/
theorem source_eq (p : Parameters) : kernel p = KldivCompute.source p := rfl

#axiomsClean source_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.KldivCompute.KernelDSL
