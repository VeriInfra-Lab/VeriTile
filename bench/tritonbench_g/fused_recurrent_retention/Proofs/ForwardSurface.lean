import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardReal
import VeriTile.Triton.Correctness

/-! Standard tagged realization and frame for both complete forward programs.
The write map follows source flags and masks; values use the initial snapshot. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardSurface
open VeriTile.Triton
noncomputable section
open scoped Classical
set_option maxHeartbeats 1800000
set_option maxRecDepth 8000

def Index (p : Layout) := (Nat × Fin p.BV) ⊕ (Fin p.BK × Fin p.BV)

def active (p : Layout) (l : Launch) (storeFinal : Bool) : Index p → Prop
  | .inl (t,j) => t < p.T ∧ activeValue p l j.val
  | .inr (k,j) => storeFinal = Bool.true ∧ activeKey p l k.val ∧ activeValue p l j.val

def address (p : Layout) (l : Launch) (r : ForwardRegions) : Index p → MemCellAddr
  | .inl (t,j) => (r.o,outputAddress p l t j.val)
  | .inr (k,j) => (r.final,stateAddress p l k.val j.val)

def writes (p : Layout) (l : Launch) (r : ForwardRegions) (storeFinal : Bool) :
    ComputeCorrect.WriteMap (Index p) :=
  ComputeCorrect.WriteMap.writeIf (active p l storeFinal) (address p l r)

def expected (p : Layout) (r : ForwardRegions) (hasSeed : Bool) (initial : BlockState) :
    Index p → MemCell
  | .inl (t,j) => .real (MaskedMath.output p (launch initial) initial r hasSeed t j)
  | .inr (k,j) => .real (MaskedMath.stateValue p (launch initial) initial r.k r.v r.initial hasSeed p.T k j)

theorem projects (p : Layout) (r : ForwardRegions) (hasSeed storeFinal swapped : Bool) :
    (ForwardSource.kernel p r hasSeed storeFinal swapped).toAlgorithm? =
      Except.ok (ForwardSource.kernel p r hasSeed storeFinal swapped).toAlgKernel := by
  cases hp : (ForwardSource.kernel p r hasSeed storeFinal swapped).toAlgorithm? with
  | ok algorithm => simp only [ComputeKernel.toAlgKernel,hp]
  | error err =>
      have hb := ForwardSource.source_body p r hasSeed storeFinal swapped
      have hempty : (ForwardSource.kernel p r hasSeed storeFinal swapped).toAlgKernel.body = [] := by
        simp only [ComputeKernel.toAlgKernel,hp]
        rfl
      rw [hempty] at hb
      have hlen := congrArg List.length hb
      simp only [List.length_nil,List.length_append,List.length_cons] at hlen
      omega

theorem realizes_of_post (kernel : ComputeKernel) (algorithm : Kernel)
    (hprojection : kernel.toAlgorithm? = Except.ok algorithm)
    (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool) (initial final : BlockState)
    (he : exec algorithm initial = some final) (hp : ForwardReal.Post p r hasSeed storeFinal initial final) :
    ComputeCorrect.RealizesFrame_without_Rounding kernel initial
      (writes p (launch initial) r storeFinal) (expected p r hasSeed initial) := by
  constructor
  · unfold writes
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgorithm_eq hprojection
    intro u v hv hu
    subst u
    rw [he] at hv
    cases hv
    intro i hi
    cases i with
    | inl row => exact hp.output row.1 row.2 hi.1 hi.2
    | inr cell => exact hp.finalState hi.1 cell.1 cell.2 hi.2.1 hi.2.2
  · apply ComputeKernel.computeCorrect_of_toAlgorithm_eq hprojection
    intro u v hv hu
    subst u
    rw [he] at hv
    cases hv
    intro cell hn
    apply hp.frame cell.1 cell.2
    · intro t j ht hj heq
      apply hn (.inl (t,j))
      simp only [writes,ComputeCorrect.WriteMap.writeIf,active,ht,hj,and_self,ite_true,address]
      exact congrArg some heq
    · intro hf k j hk hj heq
      apply hn (.inr (k,j))
      simp only [writes,ComputeCorrect.WriteMap.writeIf,active,hf,hk,hj,and_self,ite_true,address]
      exact congrArg some heq

theorem source_realizes (p : Layout) (r : ForwardRegions) (hasSeed storeFinal swapped : Bool)
    (initial : BlockState) (hqo : r.q ≠ r.o) (hko : r.k ≠ r.o) (hvo : r.v ≠ r.o)
    (hfinal : storeFinal = Bool.true → r.o ≠ r.final) :
    ComputeCorrect.RealizesFrame_without_Rounding (ForwardSource.kernel p r hasSeed storeFinal swapped)
      initial (writes p (launch initial) r storeFinal) (expected p r hasSeed initial) := by
  obtain ⟨final,he,hp⟩ := ForwardReal.source_correct p r hasSeed storeFinal swapped initial hqo hko hvo hfinal
  exact realizes_of_post _ _ (projects p r hasSeed storeFinal swapped) p r hasSeed storeFinal initial final he hp

#axiomsClean projects
#axiomsClean realizes_of_post
#axiomsClean source_realizes
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardSurface
