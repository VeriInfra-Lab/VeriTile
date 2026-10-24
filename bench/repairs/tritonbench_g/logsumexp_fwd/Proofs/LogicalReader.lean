import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.RealIO
import bench.repairs.tritonbench_g.logsumexp_fwd.IO
namespace VeriTile.Bench.Repairs.LogsumexpFwd.ReaderProof
open VeriTile.Triton FP.Structural ReaderIO
open scoped BigOperators LogicalProgramIO
noncomputable section

theorem realAt_eq (a : RawKernel.Geometry) (xs : Interface.Inputs (WithBot ℝ) a)
    (row : Fin a.rows) (column : Nat) (h : column<a.columns) :
    realAt ((inputs a) xs) row column = Interface.realInput xs (row.val*a.columns+column) := by
  have bound : row.val*a.columns+column<a.rows*a.columns := by nlinarith [row.isLt]
  unfold realAt
  rw [dif_pos h]
  have cell : ((inputs a) xs) row ⟨column,h⟩ = Interface.extend xs (row.val*a.columns+column) := by
    change xs (0 : Fin 1) (finProdFinEquiv (row,⟨column,h⟩),()) = _
    unfold Interface.extend
    rw [dif_pos bound]
    unfold Interface.value
    congr 2
    apply Fin.ext
    simp [finProdFinEquiv]
    ring
  rw [cell]
  unfold Interface.realInput
  cases Interface.extend xs (row.val*a.columns+column) <;> rfl

theorem domain_eq (a : RawKernel.Geometry) :
    (fun xs => realInputs a ((inputs a) xs)) = Interface.RealInputs a := by
  funext xs
  apply propext
  constructor
  · intro h i
    let q := finProdFinEquiv.symm i
    have hx := h q.1 q.2
    change ∃ x : ℝ, Interface.value xs i = some x
    change ∃ x : ℝ, xs (0 : Fin 1) (finProdFinEquiv q,()) = some x at hx
    simpa only [q,Equiv.apply_symm_apply,Interface.value] using hx
  · intro h row column
    exact h (finProdFinEquiv (row,column))

theorem target_eq (a : RawKernel.Geometry) (scale : ℝ) (jobs : List Nat) :
    (view a scale jobs).lift (fun xs row segment =>
      (Real.log (∑ i ∈ Finset.univ.filter
        (fun i : Fin a.block => segment.val * a.block + i.val < a.columns),
        Real.exp (if a.hasScale then
          ReaderIO.realAt xs row (segment.val * a.block + i.val) * scale
        else ReaderIO.realAt xs row (segment.val * a.block + i.val))) : WithBot ℝ)) = (fun values => Interface.result (fun j => (KernelMath.block (Interface.realInput values) (j.val / a.blocks) (j.val % a.blocks) a.columns a.block a.hasScale scale : WithBot ℝ))) := by
  funext xs o j
  fin_cases o
  let q : Fin a.rows × Fin a.blocks := finProdFinEquiv.symm j.1
  change (Real.log (∑ i ∈ Finset.univ.filter
    (fun i : Fin a.block => q.2.val*a.block+i.val<a.columns),
    Real.exp (if a.hasScale then realAt ((inputs a) xs) q.1 (q.2.val*a.block+i.val)*scale
      else realAt ((inputs a) xs) q.1 (q.2.val*a.block+i.val))) : WithBot ℝ) = _
  unfold KernelMath.block
  apply congrArg (fun value : ℝ => (Real.log value : WithBot ℝ))
  apply Finset.sum_congr rfl
  intro i hi
  rw [realAt_eq a xs q.1 _ (Finset.mem_filter.mp hi).2]
  rfl

/-- Full old/new conditional judgment equivalence. The row/block scratch
scope and ordinary-Real embedding domain are exactly the original ones. -/
theorem judgment_iff (a : RawKernel.Geometry) (scale : ℝ) (jobs : List Nat) :
    (view a scale jobs ⊨{realInputs a} fun xs row segment =>
      (Real.log (∑ i ∈ Finset.univ.filter
        (fun i : Fin a.block => segment.val * a.block + i.val < a.columns),
        Real.exp (if a.hasScale then
          ReaderIO.realAt xs row (segment.val * a.block + i.val) * scale
        else ReaderIO.realAt xs row (segment.val * a.block + i.val))) : WithBot ℝ)) ↔
      (Interface.programIO (fun _ => RealModel.engine) a (scale : WithBot ℝ) jobs).ImplementsOn
        (Interface.RealInputs a) (fun values => Interface.result (fun j => (KernelMath.block (Interface.realInput values) (j.val / a.blocks) (j.val % a.blocks) a.columns a.block a.hasScale scale : WithBot ℝ))) := by
  rw [LogicalProgramIO.implementsOn_iff,target_eq]
  change (Interface.programIO (fun _ => RealModel.engine) a (scale : WithBot ℝ) jobs).ImplementsOn
    (fun xs => realInputs a ((inputs a) xs)) _ ↔ _
  exact Iff.of_eq (congrArg
    (fun domain => (Interface.programIO (fun _ => RealModel.engine) a (scale : WithBot ℝ) jobs).ImplementsOn domain
      (fun values => Interface.result (fun j => (KernelMath.block (Interface.realInput values)
        (j.val / a.blocks) (j.val % a.blocks) a.columns a.block a.hasScale scale : WithBot ℝ))))
    (domain_eq a))

theorem correct (a : RawKernel.Geometry) (scale : ℝ) (jobs : List Nat)
    (schedule : jobs.Perm (List.range (a.rows*a.blocks))) :
    (view a scale jobs ⊨{realInputs a} fun xs row segment =>
      (Real.log (∑ i ∈ Finset.univ.filter
        (fun i : Fin a.block => segment.val * a.block + i.val < a.columns),
        Real.exp (if a.hasScale then
          ReaderIO.realAt xs row (segment.val * a.block + i.val) * scale
        else ReaderIO.realAt xs row (segment.val * a.block + i.val))) : WithBot ℝ)) :=
  (judgment_iff a scale jobs).mpr (Interface.real_implements a scale jobs schedule)
end
end VeriTile.Bench.Repairs.LogsumexpFwd.ReaderProof
