import bench.tritonbench_g.batched_vecmat_mult.Sources.Mathematical_Kernels

/-! Separate int64-address repair of the pinned Python source. The Python changes
the kernel only by integer casts: N, K and both program IDs widen before multiplication, so
`k_blocks` and the dynamic induction variable are also i64. The existing DSL
uses mathematical natural indices; its cast-erased source is reused verbatim.
`Proofs/IndexModel.lean` gives the actual fixed-width coordinate expressions separately.
The existing block-first candidate changes only the source carry addition.
The wrapper retains CUDA fp32 A[M,K], B[M,N,K], output[M,N], original allocation
and whole-block guard order, floor grid, tuning arguments and returned allocation. -/
namespace VeriTile.Bench.Repairs.BatchedVecmatMult
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
abbrev Parameters := ValueProofs.Parameters

/-- Complete source after erasing only the proven exact integer widening casts.
Inputs, products, reduction, carry, output: fp32. Native contraction and reduction
choices are kept by the separately reused graph semantics, not an FP identity. -/
abbrev kernel := TypedSource.original

def source (p : Parameters) : ComputeKernel :=
  kernel p.A p.B p.output p.M p.N p.K p.BM p.BN p.BK

/-- Existing complete candidate: the reduced block is the left operand of
the carry addition; all indexing, reduction and launch behavior is retained. -/
def blockFirst (p : Parameters) : ComputeKernel :=
  TypedSource.blockFirst p.A p.B p.output p.M p.N p.K p.BM p.BN p.BK


end VeriTile.Bench.Repairs.BatchedVecmatMult
