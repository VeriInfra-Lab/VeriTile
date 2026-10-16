import bench.tritonbench_g.rms_rbe_matmul.Proofs.Flat
import bench.tritonbench_g.rms_rbe_matmul.Proofs.Geometry
import bench.tritonbench_g.rms_rbe_matmul.Proofs.FP8Flat

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.Contracts
open VeriTile.Triton

/-- Named source-real IO: actual access safety, successful finite-flat
execution, an independent input-memory formula and the complete raw-cell frame.
This three-input batched GEMM needs its own IO surface to retain padded K reads. -/
def ExactIO (kernel : ComputeKernel) (A : FlatAlloc) (s : BlockState)
    (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat) (EPS : ℝ) : Prop :=
  kernel.eraseDType.toAlgKernel.TraceSafe A.extent s ∧
  ∃ t, exec (A.flattenKernel kernel.eraseDType.toAlgKernel) (A.flattenState s) = some t ∧
    (∀ i : TileIndex [BM, BN], active s M N BM BN i →
      t.readMem A.flat (A.addr OUT (outOffset s N BM BN sob som son i)) =
        rmsSpec s X W RMS M N BM BN sxb sxm sxk swk swn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
    (∀ r off, (r ≠ A.flat ∨ ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      off ≠ A.addr OUT (outOffset s N BM BN sob som son i)) →
      t.mem r off = (A.flattenState s).mem r off)

/-- Named byte-source IO exposes the independent supplied-array result,
actual source trace safety, successful finite-flat execution and raw frame. -/
def ByteIO (kernel : ComputeKernel) (A : FlatAlloc) (s : BlockState) (OUT : RegionName)
    (M N K sob som son BM BN BK : Nat) (EPS : ℝ)
    (xs : Fin (cdiv K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (cdiv K BK) → Fin BK → ByteExecution.Byte)
    (ws : Fin (cdiv K BK) → Fin BK → Fin BN → ByteExecution.Byte) : Prop :=
  kernel.eraseDType.toAlgKernel.TraceSafe A.extent s ∧
  ∃ t, exec (A.flattenKernel kernel.eraseDType.toAlgKernel) (A.flattenState s) = some t ∧
    (∀ i : TileIndex [BM, BN], active s M N BM BN i →
      t.readMem A.flat (A.addr OUT (outOffset s N BM BN sob som son i)) =
        PaddedRMSGemm.output K (cdiv K BK) BM BN BK EPS xs
          (fun t e => (rs t e).value) (fun t e j => (ws t e j).value) i.1 i.2.1) ∧
    (∀ r off, (r ≠ A.flat ∨ ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      off ≠ A.addr OUT (outOffset s N BM BN sob som son i)) →
      t.mem r off = (A.flattenState s).mem r off)

end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.Contracts
