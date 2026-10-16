import bench.tritonbench_g.layer_norm_fwd.Proofs.Candidate
import VeriTile.Triton.Correctness

/-! Standard region realization of the complete seven-output mathematical
contract. The write map depends only on source flags and addresses. Its values
come from the initial input snapshot, with source store order resolving output
aliases. The original and candidate use their separate successful real proofs.
Finite allocation execution is proved separately by RealIO. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.RealSurface
open VeriTile.Triton
noncomputable section
open scoped Classical

def writes (f : Flags) (r : Regions) (stride : Strides) (M N row : Nat) :
    ComputeCorrect.WriteMap MemCellAddr :=
  ComputeCorrect.WriteMap.writeIf
    (fun address => Contract.written f r stride M N row address.1 address.2) id

def expected (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (initial : BlockState) (address : MemCellAddr) : MemCell :=
  (Contract.expectedCell f r stride (snapshot r stride initial)
    M N eps p initial.pid address.1 address.2).getD (initial.mem address.1 address.2)

theorem erase_projection (kernel : ComputeKernel)
    (h : ∃ algorithm, kernel.toAlgorithm? = Except.ok algorithm) :
    kernel.eraseDType.toAlgorithm? = Except.ok kernel.eraseDType.toAlgKernel := by
  obtain ⟨algorithm, hp⟩ := h
  simp only [ComputeKernel.eraseDType, hp, ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

theorem realizes_of_contract (kernel : ComputeKernel) (algorithm : Kernel)
    (hprojection : kernel.toAlgorithm? = Except.ok algorithm)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ)
    (initial final : BlockState) (he : exec algorithm initial = some final)
    (hp : Contract.Post f r stride M N eps p initial final) :
    ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell) kernel initial
      (writes f r stride M N initial.pid) (expected f r stride M N eps p initial) := by
  constructor
  · unfold writes
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgorithm_eq hprojection
    intro s t ht hs
    subst s
    rw [he] at ht
    cases ht
    intro address _
    exact hp address.1 address.2
  · apply ComputeKernel.computeCorrect_of_toAlgorithm_eq hprojection
    intro s t ht hs
    subst s
    rw [he] at ht
    cases ht
    intro address hout
    apply Contract.post_frame f r stride M N eps p initial final hp address.1 address.2
    intro hw
    exact hout address (by simp [writes, ComputeCorrect.WriteMap.writeIf, hw])

theorem original_realizes (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (initial : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r) :
    ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
      (original f r stride M N eps p block hBlock).eraseDType initial
      (writes f r stride M N initial.pid) (expected f r stride M N eps p initial) := by
  obtain ⟨final, he, hp⟩ :=
    RealExecution.source_correct f r stride M N eps p block hBlock initial hN hsep
  exact realizes_of_contract _ _
    (erase_projection _ (original_toAlgorithm_supported f r stride M N eps p block hBlock))
    f r stride M N eps p initial final he hp

theorem candidate_realizes (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (initial : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r) :
    ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
      (candidate f r stride M N eps p block hBlock).eraseDType initial
      (writes f r stride M N initial.pid) (expected f r stride M N eps p initial) := by
  obtain ⟨final, he, hp⟩ :=
    CandidateExecution.source_correct f r stride M N eps p block hBlock initial hN hsep
  exact realizes_of_contract _ _
    (erase_projection _ (candidate_toAlgorithm_supported f r stride M N eps p block hBlock))
    f r stride M N eps p initial final he hp

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.RealSurface
