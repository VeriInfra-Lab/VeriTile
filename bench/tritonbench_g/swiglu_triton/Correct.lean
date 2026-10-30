import bench.tritonbench_g.swiglu_triton.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.SwigluTriton
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Contiguous row channels for the actual source. Forward reads A/B and
writes C; backward reads A/B/DC and overwrites A/B. The `.real` channel tag
is the Structural interpretation; the source AST retains FP32 precision. -/
abbrev swigluIO := Logical.io

/-- Pure forward value and analytic input derivatives from the original
logical snapshots. The IO masks select C in forward and DA/DB in backward. -/
def scalarFormula (ch : Channel) (a b dc : ℝ) : ℝ :=
  let sigmoid := (1 + Real.exp (-a))⁻¹
  match ch with
  | .c => a*b*sigmoid
  | .da => dc*b*sigmoid*(1+a*(1-sigmoid))
  | .db => dc*a*sigmoid

/-- Independent forward and derivative formulas on logical input channels,
with actual source access safety, successful execution, and the exterior frame. -/
specification swiglu_kernel_correct (backward : Bool) (block cols row : Nat)
    (_covers : cols ≤ block) :
    VeriTile.Spec.Real (swigluIO backward block cols row ⊨ fun xs ch i =>
      scalarFormula ch (xs.a i) (xs.b i) (xs.dc i)) :=
  Logical.reader_implements backward block cols row

#specNonCircular scalarFormula avoiding [directSource, DirectTile.sourceRun, DirectTile.scalar]
#axiomsClean swiglu_kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton
