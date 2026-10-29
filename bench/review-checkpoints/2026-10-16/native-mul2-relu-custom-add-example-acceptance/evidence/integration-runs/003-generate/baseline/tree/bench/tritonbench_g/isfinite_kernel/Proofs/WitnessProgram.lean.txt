import bench.tritonbench_g.isfinite_kernel.Proofs.WitnessOne
import bench.tritonbench_g.isfinite_kernel.Proofs.MemoryProjection

/-! Extract, rather than reconstruct, the complete source loop body. The
split equalities retain the actual branch, loop bounds, and every statement. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
open VeriTile.Triton FP.Structural TypedSource

def gridKernel32 : ComputeKernel :=
  source32 "input" "output" 1 0 2 0 14 14 2 8 Bool.false rfl rfl

def gridKernel64 : ComputeKernel :=
  source64 "input" "output" 1 0 2 0 14 14 2 8 Bool.false rfl rfl

def gridArms (k : ComputeKernel) : List ComputeStmt × List ComputeStmt :=
  match k.surfaceBody.drop 2 with
  | [.ifThenElse _ yes no] => (yes, no)
  | _ => ([], [])

def gridBody (k : ComputeKernel) : List ComputeStmt :=
  match (gridArms k).2 with
  | [_, .forRangeDyn _ _ _ _ body] => body
  | _ => []

def gridBefore : List ComputeStmt := gridKernel32.surfaceBody.take 2
def gridBody32 : List ComputeStmt := gridBody gridKernel32
def gridBody64 : List ComputeStmt := gridBody gridKernel64
def gridNumCtas : ComputeStmt := .assign .nat [] "num_ctas" (.alg (.numPrograms 0))
def gridHeader : List ComputeStmt := gridBefore ++ [gridNumCtas]

theorem grid32_split : gridKernel32.surfaceBody = gridBefore ++
    [.ifThenElse (.constBool Bool.false) (gridArms gridKernel32).1
      [gridNumCtas, .forRangeDyn "j" (.constNat 0) (.constNat 2) (.constNat 1) gridBody32]] := rfl

theorem grid64_split : gridKernel64.surfaceBody = gridBefore ++
    [.ifThenElse (.constBool Bool.false) (gridArms gridKernel64).1
      [gridNumCtas, .forRangeDyn "j" (.constNat 0) (.constNat 2) (.constNat 1) gridBody64]] := rfl

theorem grid32_body_length : gridBody32.length = 8 := rfl
theorem grid64_body_length : gridBody64.length = 8 := rfl

#axiomsClean grid32_split
#axiomsClean grid64_split
#axiomsClean grid32_body_length
#axiomsClean grid64_body_length
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
