import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Arithmetic
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
set_option maxHeartbeats 3000000

theorem instructionStep_inputs (a : Arithmetic α) (kind : Instruction) (K base : Nat)
    (xs ys xs' ys' : Nat → α) (seed : α)
    (hx : ∀ k,k<K → xs k=xs' k) (hy : ∀ k,k<K → ys k=ys' k) :
    instructionStep a kind K base xs ys seed=instructionStep a kind K base xs' ys' seed := by
  unfold instructionStep
  congr 1 <;> funext i
  · split_ifs with h
    · exact hx _ h
    · rfl
  · split_ifs with h
    · exact hy _ h
    · rfl

theorem blockStep_inputs (a : Arithmetic α) (c : Fin 98) (mode : Mode) (K block : Nat)
    (xs ys xs' ys' : Nat → α) (seed : α)
    (hx : ∀ k,k<K → xs k=xs' k) (hy : ∀ k,k<K → ys k=ys' k) :
    blockStep a c mode K block xs ys seed=blockStep a c mode K block xs' ys' seed := by
  unfold blockStep
  apply foldN_congr
  intro group seed
  exact instructionStep_inputs a _ _ _ _ _ _ _ _ hx hy

theorem accumulation_inputs (a : Arithmetic α) (c : Fin 98) (mode : Mode) (K : Nat)
    (xs ys xs' ys' : Nat → α) (bias : Option α)
    (hx : ∀ k,k<K → xs k=xs' k) (hy : ∀ k,k<K → ys k=ys' k) :
    accumulation a c mode K xs ys bias=accumulation a c mode K xs' ys' bias := by
  unfold accumulation
  apply foldN_congr
  intro block seed
  exact blockStep_inputs a _ _ _ _ _ _ _ _ _ hx hy

#axiomsClean accumulation_inputs
end
end VeriTile.Bench.Repairs.TritonLinearActivation
