import bench.tritonbench_g.log_softmax.Proofs.Graph
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.LogSoftmax
open VeriTile.Bench.TritonBenchG.LayerNormFwd.Whole
noncomputable section

theorem specificationTree_leaves (first : Fin C) (rest : List (Fin C)) :
    (specificationTree first rest).leaves=first::rest := by
  induction rest generalizing first with
  | nil=>rfl
  | cons next rest ih=>simp [specificationTree,SumTree.leaves,ih]

theorem specification_plan_exists (N : Nat) : ∃ C,Nonempty (Plan N C) := by
  have length : (List.finRange (N+1)).length=N+1 := by simp
  cases h : List.finRange (N+1) with
  | nil=>rw [h] at length; simp at length
  | cons first rest=>
    let tree:=specificationTree first rest
    have complete : tree.leaves.Perm (List.finRange (N+1)) := by
      rw [h,specificationTree_leaves]
    exact ⟨N+1,⟨⟨fun _=>tree,fun _=>tree,fun _=>tree,
      fun _=>complete,fun _=>complete,fun _=>complete,by omega,.separate⟩⟩⟩

#axiomsClean specification_plan_exists

end
end VeriTile.Bench.Repairs.LogSoftmax
