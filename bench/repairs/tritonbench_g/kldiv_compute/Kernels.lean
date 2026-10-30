import VeriTile.Triton.DSL.CallSignature
import bench.tritonbench_g.kldiv_compute.Kernels
import bench.tritonbench_g.vector_addition.Proofs.Value

/-! The repaired JIT's i64 index casts refine these exact natural indices.
The typed floating suffix is the unchanged original source. Host packing and
copyback are explicit in Proofs.Host. Installed grid-product compatibility is
outside the ordinary launch contract; no segmented-launch change is made. -/
namespace VeriTile.Bench.Repairs.KldivCompute
open VeriTile.Triton
abbrev Parameters := VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.Parameters
abbrev source (p : Parameters) :=
  VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource.kernel
    .fp32 .fp32 .fp32 Bool.false p.x p.y p.output p.n p.block
end VeriTile.Bench.Repairs.KldivCompute

namespace VeriTile.Bench.Repairs.KldivCompute.Contiguous
open VeriTile.Triton
abbrev kernel := VeriTile.Bench.TritonBenchG.KldivCompute.kldivergence_kernel
def blockSize : Nat := 1024
def gridCount (n : Nat) : Nat := (n+1023)/1024
noncomputable def mathematical (x y : ℝ) : ℝ := x * Real.log (x/y)
end VeriTile.Bench.Repairs.KldivCompute.Contiguous


namespace VeriTile.Bench.Repairs.KldivCompute.KernelDSL
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("y_ptr",.fp32),("output_ptr",.fp32)]
  scalars := [("n_elements",.integer)]
  constexpr := ["BLOCK_SIZE"]


/-- Complete selected 1024-lane source. The unchanged source uses signed32
PID/products/offsets/mask, proved exact under its existing numel < 2^31 guard.
The omitted load padding remains arbitrary and cannot reach an active store.
FP32 division, tl.log, then FP32 multiplication retain their source order. -/
def kernel (p : Parameters) : ComputeKernel :=
  let x_ptr : RegionName := p.x
  let y_ptr : RegionName := p.y
  let output_ptr : RegionName := p.output
  let BLOCK_SIZE : Nat := p.block
  with_triton_call callSignature call_args [n_elements := p.n] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * tl.log(x / y)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

/-- Existing address-commutation candidate from the original task. Only the
integer block-start operands are swapped; the FP32 computation is unchanged. -/
def commutedKernel (p : Parameters) : ComputeKernel :=
  let x_ptr : RegionName := p.x
  let y_ptr : RegionName := p.y
  let output_ptr : RegionName := p.output
  let BLOCK_SIZE : Nat := p.block
  with_triton_call callSignature call_args [n_elements := p.n] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * tl.log(x / y)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

/-- Literal typed AST equality permits reuse of the complete source proof. -/
theorem source_eq (p : Parameters) : kernel p = KldivCompute.source p := rfl

#axiomsClean source_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.KldivCompute.KernelDSL
