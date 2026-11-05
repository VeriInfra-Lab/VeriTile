import bench.tritonbench_g.iv_dependent_matmul.Proofs.Sources.Mathematical_SourceDefinitions

/-! FP32 source for all five documented induction-variable schedules. The
shared prologue, load masks, dot/update, half narrowing and FP32 store are the
existing typed source. Only the original pointer staging statements vary.
The compiled DotAcc graph remains separate from source Dot+Add. -/
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels
open VeriTile.Triton FPValue

/-- Statement boundaries in the selected complete copied body. -/
def prefixLength (schedule : Schedule) : Nat :=
  21 + match schedule with | .two => 2 | .three => 4 | _ => 0

def prologueWithPointers (schedule : Schedule) (p : Parameters) : List ComputeStmt :=
  (VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.quotation schedule .source p).surfaceBody.take
    (prefixLength schedule)

def body (schedule : Schedule) (graph : Graph) (p : Parameters) : List ComputeStmt :=
  let code : List ComputeStmt :=
    (VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.quotation schedule graph p).surfaceBody
  match code[prefixLength schedule]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ selected) => selected
  | _ => []

def loadsStart (schedule : Schedule) : Nat :=
  1 + match schedule with | .preLoad => 2 | .mixed => 1 | _ => 0

def beforeLoads (schedule : Schedule) (p : Parameters) : List ComputeStmt :=
  (body schedule .source p).take (loadsStart schedule)

def afterDot (schedule : Schedule) (p : Parameters) : List ComputeStmt :=
  (body schedule .source p).drop (loadsStart schedule+3)

/-- The complete copied program, never a reconstructed kernel authority. -/
def source := VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.quotation

theorem source_split (schedule : Schedule) (graph : Graph) (p : Parameters) :
    (source schedule graph p).surfaceBody = prologueWithPointers schedule p ++
      [.forRangeInt "k" (.constInt 0) (stopOp p) (.constNat 1) (body schedule graph p)] ++
      FPValue.suffix p := by
  cases schedule <;> cases graph <;> rfl

end VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels
