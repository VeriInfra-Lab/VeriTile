import bench.tritonbench_g.rotary_transform.Contract
import bench.tritonbench_g.rotary_transform.Proofs.TypedSetup
namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.SourceTiles
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile Triton
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000

/-- Only literal constexpr conditions are specialized. Runtime branches keep
both sides, including the early-return continuation. -/
def staticBool : Op .bool [] → Option Bool
  | .constBool b => some b
  | .boolNot b => (staticBool b).map Bool.not
  | _ => none

mutual
def algorithmShapes : Stmt → List TileShape
  | .assign _ shape _ _ => [shape]
  | .store _ shape _ _ _ => [shape]
  | .atomicAdd _ shape _ _ _ => [shape]
  | .atomicRMW _ _ shape _ _ _ _ _ => [shape]
  | .forLoop _ _ body | .forRange _ _ _ _ body | .forRangeDyn _ _ _ _ body
    | .forRangeInt _ _ _ _ body | .ifThen _ body => algorithmListShapes body
  | .ifThenElse cond left right => match staticBool cond with
    | some Bool.true => algorithmListShapes left
    | some Bool.false => algorithmListShapes right
    | none => algorithmListShapes left ++ algorithmListShapes right
def algorithmListShapes : List Stmt → List TileShape
  | [] => []
  | s::ss => algorithmShapes s ++ algorithmListShapes ss
end
mutual
def statementShapes : ComputeStmt → List TileShape
  | .alg s => algorithmShapes s
  | .assign _ shape _ _ => [shape]
  | .store _ shape _ _ _ => [shape]
  | .atomicAdd _ shape _ _ _ => [shape]
  | .atomicRMW _ _ shape _ _ _ _ _ => [shape]
  | .effectMarker _ => []
  | .forLoop _ _ body | .forRange _ _ _ _ body | .forRangeDyn _ _ _ _ body
    | .forRangeInt _ _ _ _ body | .ifThen _ body => statementListShapes body
  | .ifThenElse cond left right => match staticBool cond with
    | some Bool.true => statementListShapes left
    | some Bool.false => statementListShapes right
    | none => statementListShapes left ++ statementListShapes right
def statementListShapes : List ComputeStmt → List TileShape
  | [] => []
  | s::ss => statementShapes s ++ statementListShapes ss
end

/-- All assigned/stored tiles in the selected full quotation fit. The split
branch is checked at BM*(BK/2), the interleaved branch at BM*BK. The always
formed rk vector is also checked independently. -/
theorem quotation_shapes_bounded (s : Shape) (cu64 offset64 : Bool) (tiles : StaticTiles s) :
    ∀ shape ∈ statementListShapes (TypedExecution.kernel s cu64 offset64).surfaceBody,
      shape.prod≤1048576 := by
  have vectors := vector_bounds s tiles
  have matrix := tiles
  unfold StaticTiles matrixElements at matrix
  rw [TypedExecution.kernel,KernelSource.signed_carrier]
  cases hi : s.interleaved <;>
    simp only [hi,Bool.false_eq_true,if_false,if_true] at matrix
  all_goals cases hv : s.varlen <;> cases ho : s.offsetTensor <;> cases hc : s.conjugate <;>
    simp [KernelSource.source64_64,ComputeKernel.surfaceBody,statementListShapes,statementShapes,
      algorithmListShapes,algorithmShapes,staticBool,hi,hv,ho,hc,List.mem_append,
      vectors.1,vectors.2.1,vectors.2.2,matrix]

#axiomsClean quotation_shapes_bounded
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.SourceTiles
