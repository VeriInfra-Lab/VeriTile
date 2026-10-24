import bench.repairs.tritonbench_g.token_softmax_bloom.IO
import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.IOCorrect
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.KernelMath
namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.Logical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open SoftmaxIO
noncomputable section
/-- Coverage identifies the logical-length sum with the old complete padded output target. -/
theorem lift_eq (p : Parameters) (work : List Grid.PID) (raw : (realIO p work).Inputs)
    (admitted : realDomain p work raw) :
    (io p work).lift (fun xs job i => Real.exp (xs.values job i) /
      ∑ j : Fin (xs.length job), Real.exp (xs.at job j.val)) raw =
    target Real.algebra (fun x : ℝ => some x) p (fun _ => WithBot.unbotD 0) work realRow raw := by
  funext port
  fin_cases port
  funext i
  change Real.exp (raw (2 : Fin 3) (i.1,i.2.1,())) /
      (∑ j : Fin (length p work raw i.1), Real.exp ((inputs p work raw).at i.1 j.val)) = _
  simp only [target,realRow]
  rw [SoftmaxTriton1.RealModel.masked_sum_eq_logical _ _ (admitted.2 i.1)]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  simp only [Inputs.at,dif_pos (lt_of_lt_of_le j.isLt (admitted.2 i.1))]
  rfl

/-- Exact iff for the original complete source judgment and its unchanged real domain. -/
theorem judgment_iff (p : Parameters) (work : List Grid.PID) :
    (io p work).ImplementsOn (domain p work)
      (fun xs job i => Real.exp (xs.values job i) /
        ∑ j : Fin (xs.length job), Real.exp (xs.at job j.val)) ↔
    (realIO p work).ImplementsOn (realDomain p work)
      (target Real.algebra (fun x : ℝ => some x) p (fun _ => WithBot.unbotD 0) work realRow) := by
  have dom : (fun raw => domain p work ((io p work).inputs raw)) = realDomain p work := by
    funext raw
    change realDomain p work ((inputs p work).symm ((inputs p work) raw)) = realDomain p work raw
    exact congrArg (realDomain p work) ((inputs p work).symm_apply_apply raw)
  change (realIO p work).ImplementsOn _ _ ↔ _
  rw [dom]
  constructor
  · intro h raw admitted
    simpa only [lift_eq p work raw admitted] using h raw admitted
  · intro h raw admitted
    simpa only [lift_eq p work raw admitted] using h raw admitted
/-- The previous reader's exact guarded finite lookup, retained only to state
its complete judgment equivalence with the logical-coordinate reader. -/
def previousSegment {B : Nat} (n : Nat) (xs : Fin B → ℝ) (i : Fin B) : ℝ :=
  Real.exp (xs i) / ∑ j : Fin n,
    if hj : j.val < B then Real.exp (xs ⟨j.val,hj⟩) else 0

theorem previous_eq (p : Parameters) (work : List Grid.PID) (raw : (realIO p work).Inputs)
    (admitted : realDomain p work raw) :
    target Real.algebra (fun x : ℝ => some x) p (fun _ => WithBot.unbotD 0) work previousSegment raw =
    target Real.algebra (fun x : ℝ => some x) p (fun _ => WithBot.unbotD 0) work realRow raw := by
  funext port
  fin_cases port
  funext i
  simp only [target,previousSegment,realRow]
  rw [SoftmaxTriton1.RealModel.masked_sum_eq_logical _ _ (admitted.2 i.1)]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  simp only [dif_pos (lt_of_lt_of_le j.isLt (admitted.2 i.1))]
  rfl

/-- Direct iff with the previous reader's complete guarded target; the
canonical realRow bridge above is also the source-correctness connection. -/
theorem previous_judgment_iff (p : Parameters) (work : List Grid.PID) :
    (io p work).ImplementsOn (domain p work)
      (fun xs job i => Real.exp (xs.values job i) /
        ∑ j : Fin (xs.length job), Real.exp (xs.at job j.val)) ↔
    (realIO p work).ImplementsOn (realDomain p work)
      (target Real.algebra (fun x : ℝ => some x) p (fun _ => WithBot.unbotD 0) work previousSegment) := by
  apply (judgment_iff p work).trans
  constructor
  · intro h raw admitted
    simpa only [previous_eq p work raw admitted] using h raw admitted
  · intro h raw admitted
    simpa only [previous_eq p work raw admitted] using h raw admitted

end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.Logical
