import bench.tritonbench_g.rms_matmul_rbe.Proofs.Contract
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic
import Mathlib.Data.List.Pairwise
namespace VeriTile.Bench.Repairs.RmsMatmulRbe.Contract

theorem tensorGuard_iff (t : Tensor) (dtypes : List VeriTile.Torch.DType) :
    tensorGuard t dtypes=Bool.true ↔ TensorAdmitted t dtypes := by
  simp [tensorGuard,TensorAdmitted,and_assoc] <;> omega
theorem scalarGuard_iff (p : Scalars) (B M : Nat) (rotary : Bool) :
    scalarGuard p B M rotary=Bool.true ↔ ScalarAdmitted p B M rotary := by
  classical
  cases rotary <;> simp [scalarGuard,ScalarAdmitted,and_assoc] <;> omega
theorem readFormatGuard_iff (reads : List Tensor) :
    readFormatGuard reads=Bool.true ↔ SameReadFormat reads := by
  simp [readFormatGuard,SameReadFormat,disjoint]
theorem pythonReadFormatGuard_iff (reads : List Tensor) :
    pythonReadFormatGuard reads=Bool.true ↔ SameReadFormat reads := by
  let compatible (a b : Tensor) := (a.metadata.dtype=b.metadata.dtype ∧ a.base=b.base) ∨ disjoint a b
  have pairwise : pythonReadFormatGuard reads=Bool.true ↔ reads.Pairwise compatible := by
    induction reads with
    | nil => simp [pythonReadFormatGuard]
    | cons a tail ih => simp [pythonReadFormatGuard,ih,List.pairwise_cons,compatible,disjoint]
  rw [pairwise]
  constructor
  · intro hp
    have symm : Symmetric compatible := by
      intro a b hab
      rcases hab with h|h
      · exact Or.inl ⟨h.1.symm,h.2.symm⟩
      · exact Or.inr (by simpa only [disjoint,overlap,and_comm] using h)
    exact fun a ha b hb => hp.forall_of_forall symm (fun _ _ => Or.inl ⟨rfl,rfl⟩) ha hb
  · intro all
    clear pairwise
    induction reads with
    | nil => simp
    | cons a tail ih =>
      exact List.pairwise_cons.mpr ⟨fun b hb => all a (by simp) b (by simp [hb]),
        ih (fun x hx y hy => all x (by simp [hx]) y (by simp [hy]))⟩
theorem checked_overlap_owner (reads : List Tensor) (h : SameReadFormat reads)
    (a b : Tensor) (ha : a∈reads) (hb : b∈reads) (overlapping : overlap a b) :
    a.metadata.dtype=b.metadata.dtype ∧ a.base=b.base := by
  rcases h a ha b hb with same|apart
  · exact same
  · exact False.elim (apart overlapping)
theorem checkInputs_iff (i : Inputs) : checkInputs i=.ok () ↔ InputContract i := by
  classical
  simp [checkInputs,InputContract,ite_eq_iff,tensorGuard_iff,scalarGuard_iff,
    pythonReadFormatGuard_iff,disjoint,and_assoc]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RmsMatmulRbe.Contract
