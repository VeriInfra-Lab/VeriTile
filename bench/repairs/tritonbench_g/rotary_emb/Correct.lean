import bench.repairs.tritonbench_g.rotary_emb.Proofs.LogicalResult

namespace VeriTile.Bench.Repairs.RotaryEmb.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Each interleaved output uses its own initial cosine/sine coefficient.
The four channels below are Q-even, Q-odd, K-even and K-odd. -/
def rotation (shape : Shape) (xs : RotaryIO.RotationInputs ℝ shape)
    (key odd : Bool) (i : RotaryIO.Lane shape) : ℝ :=
  let a := xs.payload key Bool.false i
  let b := xs.payload key Bool.true i
  let c := xs.coefficient Bool.false odd i
  let z := xs.coefficient Bool.true odd i
  if odd then a*z+b*c else a*c-b*z

/-- Complete in-place Q/K rotation on the eight-input/four-output descriptor
in IO.lean. Every active source access is safe; all four output halves satisfy
the independent interleaved formula and all exterior cells are preserved. -/
specification rotary_emb_real (rest : Algebra ℝ) (shape : Shape)
    (valid : shape.Valid) (pids : Nat → Nat) :
    Spec.Real ((RotaryIO.logicalIO rest realOps shape pids) ⊨ rotation shape) :=
  (RotaryIO.logical_rotation_iff rest shape pids (rotation shape) rfl).mpr
    (RotaryIO.implements rest realOps shape valid pids)

#stmtSurfaceSubset rotary_emb_real ⊆
  [Spec.Real,FP.Structural.Algebra,Shape,Shape.Valid,RotaryIO.logicalIO,realOps,
   LogicalProgramIO.Implements,RotaryIO.RotationInputs,RotaryIO.Lane,rotation]
#specNonCircular rotation avoiding
  [Kernels.kernel,Typed.execution,Typed.reference,FP.Structural.exec]
#axiomsClean rotary_emb_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryEmb.Correct
