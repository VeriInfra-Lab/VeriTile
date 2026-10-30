import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.LogicalCoordinates

namespace VeriTile.Bench.Repairs.GegluTanhTriton
open VeriTile Triton
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Pure exact-decimal GELU approximation and its two analytic derivatives.
The three channels are C, DA and DB; the IO mask selects forward or backward. -/
def geluValues (ch : Channel) (a b dc : ℝ) : ℝ :=
  let u := 0.7978845608028654 * (a + 0.044715 * a^3)
  let gate := a / 2 * (1 + Real.tanh u)
  let slope := (1 + Real.tanh u) / 2 + a / 2 * (1 - Real.tanh u^2) *
    (0.7978845608028654 * (1 + 3 * 0.044715 * a^2))
  match ch with
  | .c => gate*b
  | .da => dc*b*slope
  | .db => dc*gate

/-- Independent forward GELU product and backward analytic derivatives over
finite input channels. Backward writes A/B in place. The shared IO judgment
proves actual source access safety, all outputs and the exact exterior frame
for either source variant and any complete legal grid order. -/
specification geglu_kernel_correctness (candidate backward : Bool) (count : Nat)
    (order : List Nat) (legal : order.Perm (List.range (Grid.grid count))) :
    Spec.Real (IO.logicalIO candidate backward count order ⊨ fun xs ch i =>
      geluValues ch (xs.a i) (xs.b i) (if backward then xs.incoming i else 0)) :=
  IO.logical_real candidate backward count order legal

#specNonCircular RealExecution.formula avoiding [source,indexCommutedSource,selectedSource,Tile.scalar]
#specNonCircular Reference.mathematical avoiding [source,indexCommutedSource,selectedSource,Tile.scalar]
#stmtSurfaceSubset geglu_kernel_correctness ⊆
  [Spec.Real,FP.Structural.LogicalProgramIO.Implements,IO.logicalIO,geluValues,Grid.grid,
   IO.TensorInputs,IO.TensorInputs.a,IO.TensorInputs.b,IO.TensorInputs.incoming,Channel]
#axiomsClean geglu_kernel_correctness
#axiomsClean Math.backward_is_gradient
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton
