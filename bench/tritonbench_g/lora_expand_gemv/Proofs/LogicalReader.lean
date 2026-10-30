import bench.tritonbench_g.lora_expand_gemv.Proofs.IOCorrect
import bench.tritonbench_g.lora_expand_gemv.IO
namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.ReaderProof
open VeriTile.Triton FP.Structural FPValue ReaderIO
open FP.Equational (ReductionPlan)
open scoped BigOperators LogicalDomainIO
noncomputable section

theorem target_eq (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (indices : Nat → Int) (data : RegionName → Nat → ℝ) (order : List Nat) :
    (view rest plans g data order).lift (fun _ row column =>
      let values := ReaderIO.observed data g
      if logicalIndex g indices row.val = -1 then values.Y row.val column.val else
        (∑ k : Fin g.K, values.X row.val k.val *
          values.W (logicalIndex g indices row.val).toNat column.val k.val) +
        (if g.add then values.Y row.val column.val else 0)) = LoraIO.realTarget data g indices := by
  funext xs o j
  fin_cases o
  rfl

/-- The exact labeled input domain and observer are retained in the complete
old/new judgment equivalence; this is not native floating-point correctness. -/
theorem judgment_iff (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (indices : Nat → Int) (data : RegionName → Nat → ℝ) (order : List Nat) :
    (view rest plans g data order ⊨{labeled g indices} fun _ row column =>
      let values := ReaderIO.observed data g
      if logicalIndex g indices row.val = -1 then values.Y row.val column.val else
        (∑ k : Fin g.K, values.X row.val k.val *
          values.W (logicalIndex g indices row.val).toNat column.val k.val) +
        (if g.add then values.Y row.val column.val else 0)) ↔
      (LoraIO.io rest plans g order).ObservesOnCells (LoraIO.realObservation data g)
        (LoraIO.domain g indices) (LoraIO.realTarget data g indices) := by
  rw [LogicalDomainIO.implementsOn_iff,target_eq]
  change (LoraIO.io rest plans g order).ObservesOnCells _
    (fun xs => (ReaderIO.inputs g).symm ((ReaderIO.inputs g) xs) = LoraIO.labeledInputs g indices) _ ↔ _
  simp only [Equiv.symm_apply_apply]
  rfl

theorem correct (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (indices : Nat → Int) (bounds : Valid g indices)
    (data : RegionName → Nat → ℝ) (order : List Nat)
    (coverage : order.Perm (List.range g.count)) :
    (view rest plans g data order ⊨{labeled g indices} fun _ row column =>
      let values := ReaderIO.observed data g
      if logicalIndex g indices row.val = -1 then values.Y row.val column.val else
        (∑ k : Fin g.K, values.X row.val k.val *
          values.W (logicalIndex g indices row.val).toNat column.val k.val) +
        (if g.add then values.Y row.val column.val else 0)) :=
  (judgment_iff rest plans g indices data order).mpr
    (IOCorrect.real rest plans g indices bounds data order coverage)
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.ReaderProof
