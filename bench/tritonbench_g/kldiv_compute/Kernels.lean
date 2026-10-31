import bench.tritonbench_g.kldiv_compute.Proofs.HistoricalKernels

namespace VeriTile.Bench.Repairs.KldivCompute.KernelDSL
open VeriTile.Triton

/-- Complete selected 1024-lane source. The unchanged source uses signed32
PID/products/offsets/mask, proved exact under its existing numel < 2^31 guard.
The omitted load padding remains arbitrary and cannot reach an active store.
FP32 division, tl.log, then FP32 multiplication retain their source order. -/
def kernel (p : Parameters) : ComputeKernel :=
  VeriTile.Bench.Repairs.KldivCompute.SelectedSource.original p.x p.y p.output p.n p.block


/-- Existing address-commutation candidate from the original task. Only the
integer block-start operands are swapped; the FP32 computation is unchanged. -/
def commutedKernel (p : Parameters) : ComputeKernel :=
  VeriTile.Bench.Repairs.KldivCompute.SelectedSource.commuted p.x p.y p.output p.n p.block


/-- Literal typed AST equality permits reuse of the complete source proof. -/
theorem source_eq (p : Parameters) : kernel p = KldivCompute.source p := rfl

#axiomsClean source_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.KldivCompute.KernelDSL
