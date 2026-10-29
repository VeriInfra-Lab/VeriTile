import bench.tritonbench_g.llama_ff_triton.Proofs.Flat

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.Contracts
open VeriTile.Triton

/- No existing KernelIO skin fits this four-input SwiGLU with ceiling-padded K
reads and both real and decoded-byte input arrays. These named IO contracts
retain actual source safety, successful finite-flat execution, the independent
array formula and the complete typed-cell frame. -/

/-- Exact source-real IO for the independent padded-array SwiGLU formula. -/
def ExactIO (kernel : ComputeKernel) (alloc : FlatAlloc) (s : BlockState)
    (OUT : RegionName) (M N K soutm soutn BM BN BK : Nat) (EPS : ℝ)
    (xs : Fin (Mathematics.steps K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (Mathematics.steps K BK) → Fin BK → ℝ)
    (w1 w3 : Fin (Mathematics.steps K BK) → Fin BK → Fin BN → ℝ) : Prop :=
  kernel.eraseDType.toAlgKernel.TraceSafe alloc.extent s ∧
  ∃ t, exec (alloc.flattenKernel kernel.eraseDType.toAlgKernel)
      (alloc.flattenState s) = some t ∧
    (∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
      t.readMem alloc.flat (alloc.addr OUT (outOffset s N BM BN soutm soutn idx)) =
        Mathematics.output K BK BM BN EPS xs rs w1 w3 idx.1 idx.2.1) ∧
    (∀ r o, (r ≠ alloc.flat ∨ ∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
      o ≠ alloc.addr OUT (outOffset s N BM BN soutm soutn idx)) →
      t.mem r o = (alloc.flattenState s).mem r o)

/-- Byte-source IO retains finite E5M2 values in the independent array formula,
actual trace safety, successful finite-flat execution and the full cell frame. -/
def ByteIO (kernel : ComputeKernel) (alloc : FlatAlloc) (s : BlockState)
    (OUT : RegionName) (M N K soutm soutn BM BN BK : Nat) (EPS : ℝ)
    (xs : Fin (Mathematics.steps K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (Mathematics.steps K BK) → Fin BK → Mathematics.Byte)
    (w1 w3 : Fin (Mathematics.steps K BK) → Fin BK → Fin BN → Mathematics.Byte) : Prop :=
  kernel.eraseDType.toAlgKernel.TraceSafe alloc.extent s ∧
  ∃ t, exec (alloc.flattenKernel kernel.eraseDType.toAlgKernel)
      (alloc.flattenState s) = some t ∧
    (∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
      t.readMem alloc.flat (alloc.addr OUT (outOffset s N BM BN soutm soutn idx)) =
        Mathematics.output K BK BM BN EPS xs (fun t e => (rs t e).value)
          (fun t e j => (w1 t e j).value) (fun t e j => (w3 t e j).value)
          idx.1 idx.2.1) ∧
    (∀ r o, (r ≠ alloc.flat ∨ ∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
      o ≠ alloc.addr OUT (outOffset s N BM BN soutm soutn idx)) →
      t.mem r o = (alloc.flattenState s).mem r o)

end VeriTile.Bench.TritonBenchG.LlamaFfTriton.Contracts
