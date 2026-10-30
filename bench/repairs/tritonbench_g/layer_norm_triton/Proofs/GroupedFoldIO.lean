import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.GroupedValues

namespace VeriTile.Bench.Repairs.LayerNormTriton.GroupedValues
open VeriTile.Triton FP.Structural SourceModel SourceGroupedGraph SourceGroupedExecution
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

def flat (a : Carries α B) : GroupedReference.Carries α B :=
  (fun j => a.1 (j,()),fun j => a.2 (j,()))

theorem fold_value (m : Model α) (s : State α) (stride M N G B group cur query : Nat)
    (x dy : Nat → Fin B → α) (w : Fin B → α) (mean inverse : Nat → α)
    (h : Inputs s stride M N B x dy w mean inverse) (a : Carries α B) :
    (((SourceGroupedExecution.fold m stride M N G B group cur s a).1.mem "DX" query).read .real,
      flat (SourceGroupedExecution.fold m stride M N G B group cur s a).2)=
      GroupedReference.fold m stride M N G B group cur query x dy w mean inverse
        ((s.mem "DX" query).read .real) (flat a) := by
  rw [SourceGroupedExecution.fold,GroupedReference.fold]
  split
  · rfl
  · split
    · rename_i hc
      rw [fold_value m (storeDX m s stride N B cur) stride M N G B group (cur+G) query
        x dy w mean inverse (after_store m s stride M N B cur x dy w mean inverse h)
        (nextCarries m s stride N B group cur a)]
      rw [store_read m s stride M N B cur query x dy w mean inverse h hc]
      have eq : flat (nextCarries m s stride N B group cur a)=
          GroupedReference.partials m N (x cur) (dy cur) (mean cur) (inverse cur)
            (decide (cur=group)) (flat a) := by
        have facts := (row_values m s stride M N B cur x dy w mean inverse h hc).2 group a
        exact Prod.ext (funext facts.1) (funext facts.2)
      rw [eq]
    · rfl
termination_by M-cur
decreasing_by omega

theorem carries_independent (m : Model α) (stride M N G B group cur query query' : Nat)
    (x dy : Nat → Fin B → α) (w : Fin B → α) (mean inverse : Nat → α)
    (prior prior' : α) (a : GroupedReference.Carries α B) :
    (GroupedReference.fold m stride M N G B group cur query x dy w mean inverse prior a).2=
      (GroupedReference.fold m stride M N G B group cur query' x dy w mean inverse prior' a).2 := by
  conv_lhs => rw [GroupedReference.fold]
  conv_rhs => rw [GroupedReference.fold]
  by_cases hg : G=0
  · simp only [hg,if_true]
  · simp only [hg,if_false]
    by_cases hc : cur<M
    · simp only [hc,if_true]
      exact carries_independent m stride M N G B group (cur+G) query query' x dy w mean inverse _ _ _
    · simp only [hc,if_false]
termination_by M-cur
decreasing_by omega

#axiomsClean fold_value
end
end VeriTile.Bench.Repairs.LayerNormTriton.GroupedValues
