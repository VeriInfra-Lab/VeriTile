import bench.tritonbench_g.rms_matmul_rbe.Proofs.QKVByteFlat
import bench.tritonbench_g.rms_matmul_rbe.Proofs.QKVFlat
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Geometry

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.Contracts
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

/-- Named complete three-output IO, including actual source trace safety,
finite-flat execution, three independent initial-input formulas and the raw
frame outside the union of the three active output footprints. -/
def QKVExactIO (kernel : ComputeKernel) (A : FlatAlloc) (s : BlockState)
    (X QW KW VW RMS Q KOut V : RegionName)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK : Nat) (EPS : ℝ) : Prop :=
  kernel.eraseDType.toAlgKernel.TraceSafe A.extent s ∧
    ∃ t, exec (A.flattenKernel kernel.eraseDType.toAlgKernel)
        (A.flattenState s) = some t ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr Q (outOffset s N BM BN sqb sqm sqn i)) =
          rmsSpec s X QW RMS M N BM BN sxb sxm sxk sqwk sqwn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr KOut (outOffset s N BM BN skb skm skn i)) =
          rmsSpec s X KW RMS M N BM BN sxb sxm sxk skwk skwn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr V (outOffset s N BM BN svb svm svn i)) =
          rmsSpec s X VW RMS M N BM BN sxb sxm sxk svwk svwn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
      (∀ r off, (r ≠ A.flat ∨ ∀ region offset,
        QKVFlat.Written s Q KOut V M N BM BN sqb sqm sqn skb skm skn svb svm svn region offset →
        off ≠ A.addr region offset) → t.mem r off = (A.flattenState s).mem r off)
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


/-- Named complete three-output IO, including actual source trace safety,
finite-flat execution, three independent initial-input formulas and the raw
frame outside the union of the three active output footprints. -/
def QKVByteIO (kernel : ComputeKernel) (A : FlatAlloc) (s : BlockState)
    (Q KOut V : RegionName)
    (M N K sqb sqm sqn skb skm skn svb svm svn BM BN BK : Nat) (EPS : ℝ)
    (xs : Fin (cdiv K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (cdiv K BK) → Fin BK → ByteExecution.Byte)
    (qws kws vws : Fin (cdiv K BK) → Fin BK → Fin BN → ByteExecution.Byte) : Prop :=
  kernel.eraseDType.toAlgKernel.TraceSafe A.extent s ∧
    ∃ t, exec (A.flattenKernel kernel.eraseDType.toAlgKernel)
        (A.flattenState s) = some t ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr Q (outOffset s N BM BN sqb sqm sqn i)) =
          PaddedRMSGemm.output K (cdiv K BK) BM BN BK EPS xs
            (fun t e => (rs t e).value) (fun t e j => (qws t e j).value) i.1 i.2.1) ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr KOut (outOffset s N BM BN skb skm skn i)) =
          PaddedRMSGemm.output K (cdiv K BK) BM BN BK EPS xs
            (fun t e => (rs t e).value) (fun t e j => (kws t e j).value) i.1 i.2.1) ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr V (outOffset s N BM BN svb svm svn i)) =
          PaddedRMSGemm.output K (cdiv K BK) BM BN BK EPS xs
            (fun t e => (rs t e).value) (fun t e j => (vws t e j).value) i.1 i.2.1) ∧
      (∀ r off, (r ≠ A.flat ∨ ∀ region offset,
        QKVFlat.Written s Q KOut V M N BM BN sqb sqm sqn skb skm skn svb svm svn region offset →
        off ≠ A.addr region offset) → t.mem r off = (A.flattenState s).mem r off)
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.Contracts
