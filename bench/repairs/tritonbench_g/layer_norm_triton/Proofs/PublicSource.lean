import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Host
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Backward
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Machine

/-! Separate repair of the complete layer_norm_triton forward/backward task.
The source AST and hashes are in Source; sourceForward/sourceBackward below are
the shared typed host entries. No reader entry is imported by a helper.

Formal precision is default FP32: loads, explicit .to(tl.float32), statistics,
partial carries, scratch, outputs and gradients are all 32-bit. Other Python
floating dtypes remain outside this claim. Shapes, strides, storage offsets,
negative-view metadata, owners, backing byte spans and aliases use existing Storage.

The normalized suffix is validated exactly, with M=product(prefix) and
N=product(normalized suffix). X/W/B and DY are independently materialized using
ordinary contiguous/view calls after exact admission. Forward saves those actual
handles, returns a view of its actual Y allocation, and retains original shape.
Empty M or N skips launches; affine gradients come from actual zero allocation.

B=min(16384,nextPowerOfTwo(N)); larger N follows the source error. Warps retain
max(min(B/256,8),1). Group count is 256/128/96/64 at N<=1024/4096/8192/otherwise.
One owner handles each modulo group in increasing row order, including the
first-without-add-zero case and every original FP32 partial cast. Scratch is
[min(G,M),N]. Final reduction retains BM=32 and BN=128 streaming order.
All raw row/column arithmetic is widened to signed64 before multiplication.

Compiler.Selected binds exact source hash, object bytes, ABI, defaults,
per-consumer reduction trees, operand swaps, FMA decisions, masks and casts.
Runtime receipts supply coordinates/status/stream/readiness, never values.
Global grid-product/PID/segmentation compatibility remains a separate boundary.
-/
namespace VeriTile.Bench.Repairs.LayerNormTriton
open VeriTile.Bench.Repairs.FastLayernorm
open VeriTile.Bench.Repairs.FastLayernorm.Host

abbrev sourceForward := @Host.forward
abbrev sourceBackward := @BackwardHost.backward
abbrev sourceAST := Source.module
abbrev originalSourceSHA256 := Source.originalSHA256
abbrev repairedSourceSHA256 := Source.candidateSHA256

def precision : String := "default FP32; all UInt32 payloads; signed64 address arithmetic"

structure Domain (x w b : Storage.Tensor) (caller : List Storage.Tensor) (initial : State Store Metadata) : Prop where
  xMember : x∈caller
  wMember : w∈caller
  bMember : b∈caller
  xLive : initial.live x
  wLive : initial.live w
  bLive : initial.live b

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.LayerNormTriton
