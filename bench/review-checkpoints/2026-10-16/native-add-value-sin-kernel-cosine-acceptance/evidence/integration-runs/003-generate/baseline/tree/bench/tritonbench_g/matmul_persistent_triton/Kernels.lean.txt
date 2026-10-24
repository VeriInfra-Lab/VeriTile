import bench.tritonbench_g.matmul_persistent_triton.Proofs.Parameters

/-!
Full persistent grouped GEMM source, revision 603e28a. Both wrapper storage
profiles retain signed negative tile/ki initialization, the exact single
k_tiles*tiles_per_SM range, original absolute-tile grouped remainder, clamped
M/N input rows/columns, K tail masks, fused dot, output cast and reset.
The candidate only commutes the signed integer output-row product.

Source-audit target: matmul_kernel_persistent
Source-audit Lean: persistentSource%
Source-audit args: dtype=tl.float16, version=original
Source-audit pointer-dtypes: {"python":{"c_ptr":"tl.float16"},"lean":{"source":"persistentSource%","args":{"dtype":"tl.float16","version":"original"}},"native":{"profile":"matmul_persistent_triton.fp16","sha256":"4325f5a352ba2e22165c482a476c32580779e3daab111bf21d08d614f6bfebba"}}
The selected textual view is the fp16 original. Both complete storage profiles
are Lean-checked and have separate pinned offline sm90 compile receipts.
All proof-side dimensions, strides and index arithmetic use unbounded Nat/Int.
Legal native Triton tile/dtype/launch configurations and fixed-width integer
intermediate correspondence are separate, unfinished admission obligations.
The two offline compile profiles do not discharge those symbolic obligations.
Textual selection does not certify native arithmetic or host execution.
-/
namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
open VeriTile.Triton

def kernel (p : Parameters) (storage : TypedSource.Storage) (v : Bool) : ComputeKernel :=
  p.source storage v

def original (p : Parameters) (storage : TypedSource.Storage) : ComputeKernel :=
  kernel p storage Bool.false

def candidate (p : Parameters) (storage : TypedSource.Storage) : ComputeKernel :=
  kernel p storage Bool.true

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
