import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardReal
import VeriTile.Triton.Correctness

/-! Standard tagged realization and frame for each complete backward source.
All expected cells are defined by standalone initial-memory gradient formulas. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSurface
open VeriTile.Triton
noncomputable section
open scoped Classical
set_option maxHeartbeats 1800000
set_option maxRecDepth 8000

inductive Index (p : Layout) where
  | dq (t : Nat) (k : Fin p.BK)
  | dk (t : Nat) (k : Fin p.BK)
  | dv (t : Nat) (j : Fin p.BV)

def active (p : Layout) (l : Launch) : Index p → Prop
  | .dq t k => t < p.T ∧ activeKey p l k.val
  | .dk t k => t < p.T ∧ activeKey p l k.val
  | .dv t j => t < p.T ∧ activeValue p l j.val

def address (p : Layout) (l : Launch) (r : BackwardRegions) : Index p → MemCellAddr
  | .dq t k => (r.dq,keyGradientAddress p l t k.val)
  | .dk t k => (r.dk,keyGradientAddress p l t k.val)
  | .dv t j => (r.dv,outputAddress p l t j.val)

def writes (p : Layout) (l : Launch) (r : BackwardRegions) : ComputeCorrect.WriteMap (Index p) :=
  ComputeCorrect.WriteMap.writeIf (active p l) (address p l r)

def expected (p : Layout) (r : BackwardRegions) (hasSeed : Bool) (initial : BlockState) :
    Index p → MemCell
  | .dq t k => .real (MaskedMath.dq p (launch initial) initial r hasSeed t k)
  | .dk t k => .real (MaskedMath.dk p (launch initial) initial r t k)
  | .dv t j => .real (MaskedMath.dv p (launch initial) initial r t j)

theorem projects (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool) :
    (BackwardSource.kernel p r hasSeed swapped).toAlgorithm? =
      Except.ok (BackwardSource.kernel p r hasSeed swapped).toAlgKernel := by
  cases hp : (BackwardSource.kernel p r hasSeed swapped).toAlgorithm? with
  | ok algorithm => simp only [ComputeKernel.toAlgKernel,hp]
  | error err =>
      have hb := BackwardSource.source_body p r hasSeed swapped
      have hempty : (BackwardSource.kernel p r hasSeed swapped).toAlgKernel.body = [] := by
        simp only [ComputeKernel.toAlgKernel,hp]
        rfl
      rw [hempty] at hb
      have hlen := congrArg List.length hb
      simp only [List.length_nil,List.length_append,List.length_cons] at hlen
      omega

theorem realizes_of_post (kernel : ComputeKernel) (algorithm : Kernel)
    (hprojection : kernel.toAlgorithm? = Except.ok algorithm)
    (p : Layout) (r : BackwardRegions) (hasSeed : Bool) (initial final : BlockState)
    (he : exec algorithm initial = some final) (hp : BackwardReal.Post p r hasSeed initial final) :
    ComputeCorrect.RealizesFrame_without_Rounding kernel initial
      (writes p (launch initial) r) (expected p r hasSeed initial) := by
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
    | dq t k => exact hp.dq t k hi.1 hi.2
    | dk t k => exact hp.dk t k hi.1 hi.2
    | dv t j => exact hp.dv t j hi.1 hi.2
  · apply ComputeKernel.computeCorrect_of_toAlgorithm_eq hprojection
    intro u v hv hu
    subst u
    rw [he] at hv
    cases hv
    intro cell hn
    apply hp.frame cell.1 cell.2
    · intro t k ht hk heq
      apply hn (.dq t k)
      simp only [writes,ComputeCorrect.WriteMap.writeIf,active,ht,hk,and_self,ite_true,address]
      exact congrArg some heq
    · intro t k ht hk heq
      apply hn (.dk t k)
      simp only [writes,ComputeCorrect.WriteMap.writeIf,active,ht,hk,and_self,ite_true,address]
      exact congrArg some heq
    · intro t j ht hj heq
      apply hn (.dv t j)
      simp only [writes,ComputeCorrect.WriteMap.writeIf,active,ht,hj,and_self,ite_true,address]
      exact congrArg some heq

theorem source_realizes (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool)
    (initial : BlockState) (hd : BackwardReal.Disjoint r) :
    ComputeCorrect.RealizesFrame_without_Rounding (BackwardSource.kernel p r hasSeed swapped) initial
      (writes p (launch initial) r) (expected p r hasSeed initial) := by
  obtain ⟨final,he,hp⟩ := BackwardReal.source_correct p r hasSeed swapped initial hd
  exact realizes_of_post _ _ (projects p r hasSeed swapped) p r hasSeed initial final he hp

#axiomsClean projects
#axiomsClean realizes_of_post
#axiomsClean source_realizes
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSurface
