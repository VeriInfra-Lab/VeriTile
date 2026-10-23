import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Partition
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Workset

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.AtomicOrder
open VeriTile.Triton Geometry
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 2000000

abbrev Orders := Workset.Orders
abbrev Event := Workset.Event
abbrev fiber := Workset.fiber

theorem fold_real {ι : Type} (xs : List ι) (f : ι → ℝ) (initial : ℝ) :
    xs.foldl (fun a i => a+f i) initial=initial+(xs.map f).sum := by
  induction xs generalizing initial with
  | nil => simp
  | cons x xs ih => simp only [List.foldl_cons,ih,List.map_cons,List.sum_cons]; ring

theorem fold_order (h : Call) (orders : Orders h) (b : Fin h.batches) (n : Fin h.N)
    (f : Fin 64 → ℝ) (initial : ℝ) :
    (orders.order b n).foldl (fun a s => a+f s) initial=initial+∑ s : Fin 64,f s := by
  rw [fold_real]
  congr 1
  rw [((orders.complete b n).map f).sum_eq]
  exact (Fin.sum_univ_def f).symm

/-- All 64 source contributions in the actual per-cell order produce the
independent Real K-term sum. Floating-point proofs cannot use this theorem. -/
theorem contribution_readback (h : Call) (orders : Orders h) (u : BlockState)
    (b : Fin h.batches) (n : Fin h.N) (slot : Nat) :
    (orders.order b n).foldl (fun a s => a+RealExecution.contribution h u b.val slot n.val s.val)
      (u.readMem "Y" (outputOffset h b.val n.val))=
    u.readMem "Y" (outputOffset h b.val n.val)+
      h.scaling*∑ k : Fin h.K,u.readMem "X" (inputOffset h b.val k.val)*
        u.readMem "W" (weightOffset h slot n.val k.val) := by
  rw [fold_order,Partition.contributions]

#axiomsClean fiber
#axiomsClean contribution_readback
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.AtomicOrder
