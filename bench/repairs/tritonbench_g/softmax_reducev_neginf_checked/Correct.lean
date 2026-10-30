import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.LogicalResult
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.KernelRealResults

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Correct
open VeriTile Triton FP.Structural
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Typed lengths, starts, page indices, logits and gathered values; one complete
(batch, head) source grid with every output channel and exact exterior frame.
The covering extent is derived from arbitrary counts, never from maximum. -/
def reduceIO (c : Host.Call) (counts : Fin c.batch → Nat) (work : List Host.Job) :=
  PublicIO.logicalSourceIO (fun _ => RealModel.model) id c (fun _ _ _ => id) counts work

/-- Independent normalized weighted sum over exactly counts[b] keys.
Negative infinity has zero mass; gathered values are finite on RealDomain. -/
def weightedSum (c : Host.Call) (counts : Fin c.batch → Nat)
    (xs : PublicIO.LogicalInputs (WithBot ℝ) c counts)
    (b : Fin c.batch) (h : Fin c.heads) (d : Fin c.dimensions) : WithBot ℝ := ↑(
    (∑ k ∈ Finset.range (counts b),
      (xs.logitAt b h k).elim 0 Real.exp * (xs.valueAt b h k d).unbotD 0) /
    (∑ k ∈ Finset.range (counts b), (xs.logitAt b h k).elim 0 Real.exp))

/-- Independent normalized weighted sums. Each active row has a finite logit
and finite gathered values; negative-infinity logits have zero mass. Actual
source access safety, success and all outputs are part of the shared judgment. -/
specification kernel_real (c : Host.Call) (counts : Fin c.batch → Nat)
    (work : List Host.Job) (complete : work.Perm (Host.workset c)) :
    Spec.Real ((reduceIO c counts work) ⊨{PublicIO.LogicalRealDomain} weightedSum c counts) :=
  (PublicIO.logical_real_iff c counts work).mpr (PublicIO.real_implements c counts work complete)

#axiomsClean kernel_real
#stmtSurfaceSubset kernel_real ⊆
  [Spec.Real,LogicalProgramIO.ImplementsOn,reduceIO,PublicIO.LogicalRealDomain,weightedSum,
   Host.Call,Host.Call.batch,PublicIO.LogicalInputs,Host.Call.heads,Host.Call.dimensions,Host.Job,Host.workset,List.Perm]
#specNonCircular weightedSum avoiding
  [FPValue.source,FPValue.outputValues,FPValue.finalAccumulator,RawKernel.runGrid,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Correct
