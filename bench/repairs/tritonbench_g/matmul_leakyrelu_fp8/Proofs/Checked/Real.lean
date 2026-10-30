import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Checked.Source

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
noncomputable section
open scoped BigOperators
set_option maxHeartbeats 600000

def realOps : Operations ℝ where
  zero := 0
  castHalf := id
  nonnegative := fun x => decide (0 ≤ x)
  slopeMultiply := fun x => (1/100 : ℝ)*x
  instruction := fun xs ys seed => seed + ∑ i, xs i*ys i

theorem fold_add (items : List β) (f : β → ℝ) (z : ℝ) :
    items.foldl (fun z i => z+f i) z = z+(items.map f).sum := by
  induction items generalizing z with
  | nil => simp
  | cons i rest ih => simp only [List.foldl_cons,ih,List.map_cons,List.sum_cons]; ring

theorem block_real (xs ys : Fin 64 → ℝ) (seed : ℝ) :
    block realOps xs ys seed = (∑ i, xs i*ys i)+seed := by
  unfold block
  simp only [realOps]
  rw [fold_add,←Fin.sum_univ_def]
  have partition : (∑ group : Fin 8, ∑ lane : Fin 8,
      xs ⟨group.val*8+lane.val,by omega⟩*ys ⟨group.val*8+lane.val,by omega⟩) = ∑ i, xs i*ys i := by
    have partition := Fintype.sum_equiv (finProdFinEquiv : Fin 8 × Fin 8 ≃ Fin 64)
      (fun pair => xs ⟨pair.1.val*8+pair.2.val,by omega⟩*ys ⟨pair.1.val*8+pair.2.val,by omega⟩)
      (fun i => xs i*ys i) (fun pair => by
        have coordinate : (⟨pair.1.val*8+pair.2.val,by omega⟩ : Fin 64) = finProdFinEquiv pair := by
          apply Fin.ext
          simp [finProdFinEquiv,Nat.mul_comm,Nat.add_comm]
        exact congrArg (fun k => xs k*ys k) coordinate)
    simpa only [Fintype.sum_prod_type] using partition
  rw [partition]
  ring

theorem activate_real (enabled : Bool) (value : ℝ) :
    activate realOps enabled value = Real.activate enabled value := by
  cases enabled <;> simp [activate,Real.activate,realOps]

theorem accumulator_real (p : Problem) (b : Buffers) (memory : Memory ℝ)
    (tile count : Nat) :
    Source.accumulator (sourceArithmetic realOps) p b memory tile count =
      Source.accumulator Real.arithmetic p b memory tile count := by
  have aa : ∀ i, Source.loadA (sourceArithmetic realOps) p b memory tile i = Source.loadA Real.arithmetic p b memory tile i := by
    intro i; rfl
  have bb : ∀ i, Source.loadB (sourceArithmetic realOps) p b memory tile i = Source.loadB Real.arithmetic p b memory tile i := by
    intro i; rfl
  induction count with
  | zero => rfl
  | succ count ih =>
    funext row column
    change block realOps (Source.loadA (sourceArithmetic realOps) p b memory tile count row)
      (fun k => Source.loadB (sourceArithmetic realOps) p b memory tile count k column)
      (Source.accumulator (sourceArithmetic realOps) p b memory tile count row column) =
      (∑ k, Source.loadA Real.arithmetic p b memory tile count row k *
        Source.loadB Real.arithmetic p b memory tile count k column) +
      Source.accumulator Real.arithmetic p b memory tile count row column
    rw [block_real,aa,bb,ih]
    rfl

theorem source_value_real (p : Problem) (b : Buffers) (memory : Memory ℝ) (tile : Nat) (point : Address) :
    Source.value (sourceArithmetic realOps) p b memory tile point = Source.value Real.arithmetic p b memory tile point := by
  unfold Source.value
  split_ifs
  · change activate realOps p.enabled (Source.accumulator (sourceArithmetic realOps) p b memory tile p.kTiles _ _) =
      Real.activate p.enabled (Source.accumulator Real.arithmetic p b memory tile p.kTiles _ _)
    rw [accumulator_real,activate_real]
  · rfl

theorem source_run_real (p : Problem) (b : Buffers) (memory : Memory ℝ) :
    Source.run (sourceArithmetic realOps) p b memory = Source.run Real.arithmetic p b memory := by
  have steps : Source.step (sourceArithmetic realOps) p b = Source.step Real.arithmetic p b := by
    funext tile m point
    simp only [Source.step,CrossEntropy1.Schedule.put,source_value_real]
  simp only [Source.run,steps]

#axiomsClean block_real
#axiomsClean source_run_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
