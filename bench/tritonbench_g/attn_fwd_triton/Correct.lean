import bench.tritonbench_g.attn_fwd_triton.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked
open VeriTile Triton AttnFwdTriton Verified
open scoped BigOperators VeriTile.Triton.LogicalIO
noncomputable section

/-- Block scales are indexed by batch, head and their query/key block. -/
def attentionScore (p : RealProofs.Parameters) (xs : RealInterface.LogicalInputs p)
    (batch head row key : Nat) : ℝ :=
  ((List.range p.headDim).map (fun d =>
    xs.tensors.query batch head row d * xs.tensors.key batch head key d)).sum *
    xs.queryScaleAt batch head (row/128) * xs.keyScaleAt batch head (key/64)

/-- Independent causal scaled-softmax attention at batch/head/row/feature
coordinates, with all five strided input roles and their allowed aliases. -/
def mathematical (p : RealProofs.Parameters) (values : RealInterface.LogicalInputs p)
    (c : RealInterface.Coordinate p) : ℝ :=
  let xs := values.tensors
  let pairs := (List.range (c.2.2.1.val+1)).map (fun n =>
    (attentionScore p values c.1.val c.2.1.val c.2.2.1.val n,
      xs.value c.1.val c.2.1.val n c.2.2.2.val))
  (pairs.map (fun sv => Real.exp (Real.log 2 * sv.1) * sv.2)).sum /
    (pairs.map (fun sv => Real.exp (Real.log 2 * sv.1))).sum

/-- Complete causal grid, successful source execution, actual source safety
and exterior frame. Source arithmetic precision remains fixed by Kernels. -/
specification whole_real (p : RealProofs.Parameters) (A : FlatAlloc)
    (stage : p.stage=3) (nonoverlap : OutputLayout.NonOverlapping p) :
    Spec.Real (RealInterface.attentionIO A p ⊨ mathematical p) :=
  RealInterface.logical_implements A p stage nonoverlap

#axiomsClean whole_real
#specNonCircular mathematical avoiding [exec,execR,AlgorithmExecution,Kernel.GridLaunchedOrdinary]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked
