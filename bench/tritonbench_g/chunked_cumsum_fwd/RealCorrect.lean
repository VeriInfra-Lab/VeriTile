import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.LogicalIO

/-! Full-grid kernel IO and independent preparation/prefix mathematics.
IO.lean exposes input/output shapes, strides and runtime scalar bindings;
Proofs/ contains actual-source safety, execution and exact frame assembly. -/
namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Correct
open VeriTile Triton FP.Structural
open Kernel Kernel.Scalar
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
open scoped BigOperators
noncomputable section

/- Independent scalar mathematics and the complete logical output channels.
These definitions are definitionally connected to the canonical target used below. -/
namespace Mathematics

def softplus (x : ℝ) : ℝ := if x ≤ 20 then Real.log (1+Real.exp x) else x

def prepared (c : Geometry) (a : LogicalInputs ℝ) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (b h k j : Nat) : ℝ :=
  if h < c.heads ∧ j < c.chunk ∧ k*c.chunk+j < c.sequence then
    let x := a.dt b (k*c.chunk+j) h + if hasBias then a.bias h else 0
    upper.cap (max (if useSoftplus then softplus x else x) lower)
  else 0

def cumulative (c : Geometry) (a : LogicalInputs ℝ) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (b h k j : Nat) : ℝ :=
  ∑ t ∈ Finset.range (j+1), prepared c a hasBias useSoftplus lower upper b h k t * a.rate h
end Mathematics


/-- The complete grid consumes dt/rate and optional bias, then writes both
packed result arrays. The two clamp values are runtime scalar registers. -/
def cumsumIO (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (hasBias reverseBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) :=
  LogicalTensors.io c plan hasBias reverseBias useSoftplus lower upper

/-- Independent softplus/clamp preparation and inclusive sum of prepared*rate.
Padding stores zero preparation and continues the inclusive cumulative sum. -/
specification kernel_correct (c : Geometry) (sizes : Sizes c) (plan : PrefixPlan c.BH c.BC)
    (hasBias reverseBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) :
    Spec.Real (cumsumIO c plan hasBias reverseBias useSoftplus lower upper ⊨ fun values =>
      { prepared := fun b h chunk lane => Mathematics.prepared c
          (LogicalTensors.inputArrays c values) hasBias useSoftplus lower upper b.val h.val chunk.val lane.val
        cumulative := fun b h chunk lane => Mathematics.cumulative c
          (LogicalTensors.inputArrays c values) hasBias useSoftplus lower upper b.val h.val chunk.val lane.val }) := by
  exact LogicalTensors.correct c plan hasBias reverseBias useSoftplus lower upper sizes

#specNonCircular Mathematics.prepared avoiding [Kernel.launch, Scalar.interpret]
#specNonCircular Mathematics.cumulative avoiding [Kernel.launch, Scalar.interpret]
#axiomsClean kernel_correct
#stmtSurfaceSubset kernel_correct ⊆
  [cumsumIO, LogicalTensors.inputArrays, LogicalTensors.Outputs.mk,LogicalTensors.Outputs,Geometry.batches,Geometry.heads,Geometry.chunks,Geometry.chunk, Geometry, Geometry.BC, Geometry.BH,
   PrefixPlan, Mathematics.cumulative, Mathematics.prepared, Term, instInhabitedTerm, LogicalTensors.Inputs,
   Sizes, UpperBound, Spec.Real, LogicalProgramIO.Implements]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Correct
