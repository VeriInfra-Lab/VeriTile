import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealExecution
import VeriTile.Triton.Correctness

/-! The standard successful-projection realization and full tagged frame
for both complete original and candidate scalar kernels. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealSurface
open VeriTile.Triton RealValues RealLoop
noncomputable section
open scoped Classical

def writes (T row : Nat) (o : RegionName) : ComputeCorrect.WriteMap Nat :=
  ComputeCorrect.WriteMap.writeIf (fun t => t < T) (fun t => (o, address T row t))

def expectedCell (T BT : Nat) (s : RegionName) (initial : BlockState) (t : Nat) : MemCell :=
  MemCell.real (expected T BT (initial.pids 0) (initial.readMem s) (initial.undef s) t)

theorem erase_projection (kernel : ComputeKernel)
    (h : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel) :
    kernel.eraseDType.toAlgorithm? = Except.ok kernel.eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, h, ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

theorem realizes_of_post (kernel : ComputeKernel) (algorithm : Kernel)
    (hprojection : kernel.toAlgorithm? = Except.ok algorithm)
    (T BT : Nat) (s o : RegionName) (initial final : BlockState)
    (he : exec algorithm initial = some final) (hp : RealExecution.Post T BT s o initial final) :
    ComputeCorrect.RealizesFrame_without_Rounding kernel initial
      (writes T (initial.pids 0) o) (expectedCell T BT s initial) := by
  constructor
  · unfold writes
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgorithm_eq hprojection
    intro u v hv hu
    subst u
    rw [he] at hv
    cases hv
    intro t ht
    exact hp.1 t ht
  · apply ComputeKernel.computeCorrect_of_toAlgorithm_eq hprojection
    intro u v hv hu
    subst u
    rw [he] at hv
    cases hv
    intro cell hn
    apply (hp.2 cell.1 cell.2 ?_).symm
    rintro ⟨hr,t,ht,hoff⟩
    apply hn t
    simp only [writes, ComputeCorrect.WriteMap.writeIf, if_pos ht, Option.some.injEq]
    exact Prod.ext hr.symm hoff.symm

theorem source_realizes (swapped : Bool) (T BT : Nat) (s o : RegionName)
    (dtype out_dtype : FloatDType) (initial : BlockState) (hSO : s ≠ o) (hBT : 0 < BT) :
    ComputeCorrect.RealizesFrame_without_Rounding
      (TypedSource.source swapped dtype out_dtype s o T BT).eraseDType initial
      (writes T (initial.pids 0) o) (expectedCell T BT s initial) := by
  obtain ⟨final,he,hp⟩ := RealExecution.source_correct swapped T BT s o dtype out_dtype initial hSO hBT
  exact realizes_of_post _ _ (erase_projection _ (TypedSource.projects ..)) T BT s o initial final he hp

#axiomsClean realizes_of_post
#axiomsClean source_realizes
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealSurface
