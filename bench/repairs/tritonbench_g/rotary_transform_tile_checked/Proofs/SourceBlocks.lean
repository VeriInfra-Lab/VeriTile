import bench.repairs.tritonbench_g.rotary_transform_tile_checked.KernelSource
import bench.repairs.tritonbench_g.rotary_transform.Spec
import VeriTile.Triton.Float.Structural
namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.SourceBlocks
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 1000000
set_option maxRecDepth 24000

/-- Proof-only specialization of the source's immutable varlen branch.
Canonical kernels remain complete raw quotations in Kernels.lean. -/
def select (flag : Bool) : ComputeStmt → List ComputeStmt
  | .ifThenElse _ yes no => if flag then no else yes
  | _ => []
def body (flag : Bool) (k : ComputeKernel) : List ComputeStmt :=
  k.surfaceBody.take 4 ++ (k.surfaceBody.drop 4).flatMap (select flag)
def setupCount (varlen : Bool) : Nat := if varlen then 8 else 6

theorem run_constant_branch {α : Type} [Inhabited α] (M : Algebra α)
    (flag : Bool) (prelude yes no : List ComputeStmt) (s : State α) :
    run M (prelude ++ [.ifThenElse (.boolNot (.constBool flag)) yes no]) s =
      run M (prelude ++ if flag then no else yes) s := by
  simp only [run_append]
  congr 1
  funext t
  cases flag <;> simp [run,step,evalOp_unfold]

#axiomsClean run_constant_branch
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.SourceBlocks
