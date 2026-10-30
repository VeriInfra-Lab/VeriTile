import bench.tritonbench_g.bgmv_expand_slice.Proofs.IOBindings

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicIO
open VeriTile.Triton FP.Structural Direct Direct.FPValues
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

theorem value_binding (M : Algebra A) (R : Reduction A) (project : A → A)
    (h : Call) (xs : Inputs A h) (s : State A)
    (hin : (sourceIO M R project h).InputBinding xs s)
    (batch : Fin h.batches) (row : Fin h.N) (active : index xs batch≠-1) :
    FPGrid.value M R h ⟨batch.val,row.val,batch.isLt,row.isLt⟩ (PublicExecution.data s)=
      localValue M R h (input xs batch) (weight xs batch row) (previous xs batch row) := by
  have positive : 0<h.N := by have := row.isLt;omega
  have hi := index_binding M R project h xs s hin positive batch
  have productsEq : Direct.FPValues.products M h batch.val
      ((PublicExecution.data s).indices batch.val).toNat row.val (PublicExecution.data s)=
      (fun j : TileIndex [256,h.BK] => M.binary (some .fp32) .fp16 .mul
        (M.cast (some .fp32) .real .fp16 (M.fp32Load
          (if h.even=Bool.true ∨ j.2.1.val<h.K then input xs batch j.2.1.val
            else M.literal (some .fp32) .real 0)))
        (if j.1.val<1 ∧ j.2.1.val<h.K then weight xs batch row j.2.1.val
          else M.cast (some .fp32) .real .fp16 (M.literal (some .fp32) .real 0.0))) := by
    funext j
    rcases j with ⟨n,k,u⟩
    simp only [Direct.FPValues.products,Direct.FPValues.loaded,Direct.FPValues.weight]
    congr 1
    · congr 1
      congr 1
      split_ifs with access
      · apply x_binding M R project h xs s hin batch positive active
        unfold inputWidth
        cases he : h.even
        · simp only [he,Bool.false_eq_true,↓reduceIte]
          exact access.resolve_left (by simp [he])
        · simp only [he,↓reduceIte]
          exact k.isLt
      · rfl
    · split_ifs with access
      · have zero : n.val=0 := by omega
        simp only [zero,Nat.zero_mul,Nat.add_zero,hi]
        exact weight_binding M R project h xs s hin batch row active k.val access.2
      · rfl
  unfold FPGrid.value Direct.FPValues.result Direct.FPValues.sums localValue
  rw [productsEq]
  cases adds : h.add
  · rfl
  · simp only [adds,↓reduceIte,Direct.FPValues.old,Fin.val_zero,Nat.zero_lt_one,
      Nat.add_zero]
    rw [previous_binding M R project h xs s hin batch row active adds]
    rfl

#axiomsClean value_binding
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicIO
