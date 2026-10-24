import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.Correct
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
open scoped BigOperators LogicalDomainIO
noncomputable section

/-- Complete dense GEMV, sentinel preservation and optional prior-Y addition.
Real payloads are the original address-label observation, explicitly distinct
from the exact typed graph-label source inputs in the public labeled domain. -/
specification kernel_real (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (indices : Nat → Int) (bounds : Valid g indices)
    (data : RegionName → Nat → ℝ) (order : List Nat)
    (coverage : order.Perm (List.range g.count)) :
    Spec.Real (ReaderIO.view rest plans g data order ⊨{ReaderIO.labeled g indices} fun _ row column =>
      let values := ReaderIO.observed data g
      if logicalIndex g indices row.val = -1 then values.Y row.val column.val else
        (∑ k : Fin g.K, values.X row.val k.val *
          values.W (logicalIndex g indices row.val).toNat column.val k.val) +
        (if g.add then values.Y row.val column.val else 0)) :=
  ReaderProof.correct rest plans g indices bounds data order coverage

#axiomsClean kernel_real
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.Correct
