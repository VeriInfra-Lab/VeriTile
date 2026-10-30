import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.IO
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.IOReal

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
open VeriTile Triton FP.Structural
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Complete original source-grid judgment, reindexed bijectively. Padded and
inactive values, starts/pages, normalization domain, work order and frame remain. -/
theorem logical_real_iff (c : Host.Call) (counts : Fin c.batch → Nat) (work : List Host.Job) :
    (logicalSourceIO (fun _ => RealModel.model) id c (fun _ _ _ => id) counts work
      ⊨{LogicalRealDomain} fun xs b h d => ↑(
        (∑ k ∈ Finset.range (counts b), (xs.logitAt b h k).elim 0 Real.exp *
          (xs.valueAt b h k d).unbotD 0) /
        (∑ k ∈ Finset.range (counts b), (xs.logitAt b h k).elim 0 Real.exp))) ↔
    (realIO c counts work).ImplementsOn RealDomain (mathematical c counts) := by
  have target_eq : (logicalSourceIO (fun _ => RealModel.model) id c (fun _ _ _ => id) counts work).lift
      (fun xs b h d => ↑(
        (∑ k ∈ Finset.range (counts b), (xs.logitAt b h k).elim 0 Real.exp *
          (xs.valueAt b h k d).unbotD 0) /
        (∑ k ∈ Finset.range (counts b), (xs.logitAt b h k).elim 0 Real.exp))) =
      mathematical c counts := by
    funext raw port j
    fin_cases port
    rfl
  change (realIO c counts work).ImplementsOn RealDomain _ ↔ _
  rw [target_eq]

#axiomsClean logical_real_iff
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
