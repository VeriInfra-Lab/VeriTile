import bench.tritonbench_g.reversed_cumsum.Proofs.RealExecution
import VeriTile.Triton.Correctness

/-! Standard successful-projection realization and complete tagged frame for
the actual original and candidate kernels. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.RealSurface
open VeriTile.Triton RealProgram RealStep RealLoop
noncomputable section
open scoped Classical

def writes (p : Layout) (l : Launch) (o : RegionName) :
    ComputeCorrect.WriteMap (Nat × Fin p.BS) :=
  ComputeCorrect.WriteMap.writeIf (fun i => active p l i.1 i.2.val)
    (fun i => (o, address p l i.1 i.2.val))

def expectedCell (p : Layout) (s : RegionName) (initial : BlockState)
    (i : Nat × Fin p.BS) : MemCell :=
  MemCell.real (expected p (RealInit.launch initial) (initial.readMem s) (initial.undef s) i.1 i.2)

theorem erase_projection (kernel : ComputeKernel)
    (h : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel) :
    kernel.eraseDType.toAlgorithm? = Except.ok kernel.eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, h, ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

theorem realizes_of_post (kernel : ComputeKernel) (algorithm : Kernel)
    (hprojection : kernel.toAlgorithm? = Except.ok algorithm)
    (p : Layout) (s o : RegionName) (initial final : BlockState)
    (he : exec algorithm initial = some final) (hp : RealExecution.Post p s o initial final) :
    ComputeCorrect.RealizesFrame_without_Rounding kernel initial
      (writes p (RealInit.launch initial) o) (expectedCell p s initial) := by
  constructor
  · unfold writes
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgorithm_eq hprojection
    intro u v hv hu
    subst u
    rw [he] at hv
    cases hv
    intro i hi
    exact hp.1 i.1 i.2 hi
  · apply ComputeKernel.computeCorrect_of_toAlgorithm_eq hprojection
    intro u v hv hu
    subst u
    rw [he] at hv
    cases hv
    intro cell hn
    apply (hp.2 cell.1 cell.2 ?_).symm
    rintro ⟨hr, t, j, ha, hoff⟩
    apply hn (t,j)
    simp only [writes, ComputeCorrect.WriteMap.writeIf, if_pos ha, Option.some.injEq]
    exact Prod.ext hr.symm hoff.symm

theorem source_realizes (swapped : Bool) (p : Layout) (s o : RegionName)
    (dtype out_dtype : FloatDType) (initial : BlockState)
    (hSO : s ≠ o) (hBT : 0 < p.BT) (hinj : NoCollision p (RealInit.launch initial)) :
    ComputeCorrect.RealizesFrame_without_Rounding
      (RealSource.source swapped dtype out_dtype s o p.headStride p.timeStride p.channelStride
        p.T p.S p.BT p.BS).eraseDType initial
      (writes p (RealInit.launch initial) o) (expectedCell p s initial) := by
  obtain ⟨final, he, hp⟩ := RealExecution.source_correct swapped p s o dtype out_dtype initial hSO hBT hinj
  exact realizes_of_post _ _ (erase_projection _ (RealSource.projects ..)) p s o initial final he hp

#axiomsClean realizes_of_post
#axiomsClean source_realizes
end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.RealSurface
