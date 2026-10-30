import bench.tritonbench_g.destindex_copy_kv1.IO
import bench.tritonbench_g.destindex_copy_kv1.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO
open VeriTile.Triton FP.Structural
noncomputable section

theorem target_transport (α : Type) (c : Config) :
    (fun raw => (outputEquiv α c).symm (fun row head dim =>
      (inputEquiv α c raw).payload row head dim)) = target c := by
  funext raw o j
  fin_cases o
  rcases j with ⟨row,head,dim,u⟩
  cases u
  rfl

theorem admissible_pullback (α : Type) (c : Config) :
    (fun raw => LogicalAdmissible c (inputEquiv α c raw)) = Admissible c := rfl

/-- The entire previous contract, with the same domain, execution, bounds and frame. -/
theorem logical_iff (rest : Algebra ℝ) (c : Config) (ps : List Nat) :
    (logicalIO rest c ps).ImplementsOn (LogicalAdmissible c)
      (fun xs row head dim => xs.payload row head dim) ↔
      (copyIO rest c ps).ImplementsOn (Admissible c) (target c) := by
  change (copyIO rest c ps).ImplementsOn
    (fun raw => LogicalAdmissible c (inputEquiv ℝ c raw))
    (fun raw => (outputEquiv ℝ c).symm (fun row head dim =>
      (inputEquiv ℝ c raw).payload row head dim)) ↔ _
  exact Iff.of_eq (congrArg₂ (fun domain formula =>
    (copyIO rest c ps).ImplementsOn domain formula)
      (admissible_pullback ℝ c) (target_transport ℝ c))

theorem logical_correct (rest : Algebra ℝ) (c : Config) (ps : List Nat)
    (order : ∀ p,p∈ps ↔ p<c.count) :
    (logicalIO rest c ps).ImplementsOn (LogicalAdmissible c)
      (fun xs row head dim => xs.payload row head dim) :=
  (logical_iff rest c ps).mpr (implements rest c ps order)

end
end VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO
