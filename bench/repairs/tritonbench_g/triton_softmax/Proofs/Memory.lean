import bench.repairs.tritonbench_g.triton_softmax.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.TritonSoftmax
/-- A fold whose writes depend only on the observed memory cannot observe
temporary registers in its initial state. -/
theorem fold_observe {A B I : Type} (observe : A → B) (step : A → I → A)
    (items : List I) (s t : A) (same : observe s=observe t)
    (respects : ∀ s t i,observe s=observe t → observe (step s i)=observe (step t i)) :
    observe (items.foldl step s)=observe (items.foldl step t) := by
  induction items generalizing s t with
  | nil => exact same
  | cons i tail ih => exact ih (step s i) (step t i) (respects s t i same)

#axiomsClean fold_observe
end VeriTile.Bench.Repairs.TritonSoftmax
