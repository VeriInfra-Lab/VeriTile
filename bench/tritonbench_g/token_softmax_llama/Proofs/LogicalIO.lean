import bench.tritonbench_g.token_softmax_llama.IO
import bench.tritonbench_g.token_softmax_llama.Proofs.LogicalColumns

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.PublicIO
open VeriTile.Triton FP.Structural Strided
open TokenSoftmaxBloom.SoftmaxIO (realDomain target realRow length)
noncomputable section

def logicalFormula (p : Parameters) (work : List Grid.PID) (xs : LogicalInputs p work)
    (r : Fin work.length) (c : Fin p.width) : ℝ :=
  Real.exp (xs.values r c) / ∑ j : Fin (xs.lengths r),Real.exp (xs.segment p work r j)

theorem lift_eq (p : Parameters) (work : List Grid.PID) (xs : (realIO p work).Inputs)
    (domain : realDomain p work xs) :
    (logicalRealIO p work).lift (logicalFormula p work) xs =
      target Real.algebra (fun x : ℝ => some x) p (fun _ => WithBot.unbotD 0) work realRow xs := by
  funext port i
  fin_cases port
  change Real.exp (xs (2 : Fin 3) i) /
      (∑ j : Fin (length p work xs i.1),Real.exp
        (if h : j.val < p.width then xs (2 : Fin 3) (i.1,⟨j.val,h⟩,()) else 0)) =
    Real.exp (xs (2 : Fin 3) i) /
      ∑ j : Fin p.width,if j.val < length p work xs i.1 then Real.exp (xs (2 : Fin 3) (i.1,j,())) else 0
  rw [LogicalColumns.sum_eq _ _ (domain.2 i.1)]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  have hj : j.val < p.width := lt_of_lt_of_le j.isLt (domain.2 i.1)
  simp only [LogicalInputs.segment,logicalInputs,hj,dif_pos]

theorem logical_iff (p : Parameters) (work : List Grid.PID) :
    (logicalRealIO p work).ImplementsOn (logicalDomain p work) (logicalFormula p work) ↔
      (realIO p work).ImplementsOn (realDomain p work)
        (target Real.algebra (fun x : ℝ => some x) p (fun _ => WithBot.unbotD 0) work realRow) := by
  change (realIO p work).ImplementsOn _ _ ↔ _
  have hd : (fun xs => logicalDomain p work ((logicalRealIO p work).inputs xs)) = realDomain p work := by
    funext xs
    exact congrArg (realDomain p work) ((logicalInputs p work).symm_apply_apply xs)
  rw [hd]
  unfold ProgramIO.ImplementsOn
  apply forall_congr'; intro xs
  apply forall_congr'; intro domain
  rw [lift_eq p work xs domain]
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.PublicIO
