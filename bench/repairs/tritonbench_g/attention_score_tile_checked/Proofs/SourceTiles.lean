import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked
open VeriTile Triton

mutual
def algorithmShapes : Stmt → List TileShape
  | .assign _ shape _ _ => [shape]
  | .store _ shape _ _ _ => [shape]
  | .atomicAdd _ shape _ _ _ => [shape]
  | .atomicRMW _ _ shape _ _ _ _ _ => [shape]
  | .forLoop _ _ body | .forRange _ _ _ _ body | .forRangeDyn _ _ _ _ body
    | .forRangeInt _ _ _ _ body | .ifThen _ body => algorithmListShapes body
  | .ifThenElse _ left right => algorithmListShapes left ++ algorithmListShapes right
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
  | .ifThenElse _ left right => statementListShapes left ++ statementListShapes right
def statementListShapes : List ComputeStmt → List TileShape
  | [] => []
  | s::ss => statementShapes s ++ statementListShapes ss
end

/-- Read the actual quotation's assigned/store shapes, through both branches
and the loop. Internal broadcast vectors fit the separately listed B/1/D
shapes; the Q/K block-pointer assignments carry the largest B*D shape. -/
theorem quotation_shapes_bounded
    (Q K M O : RegionName) (sqz sqh sqm sqk skz skh skn skk soz soh son
      Z H HK N R NK B D : Nat) (off sz : Int) (scale : ℝ)
    (window complement evenM evenN : Bool) (smallBlock : B≤64) (tile : B*D≤1048576) :
    ∀ shape ∈ statementListShapes
      (Reused.Whole002.TypedSource.widened Q K M O sqz sqh sqm sqk skz skh skn skk soz soh son
        Z H HK N R NK B D B off sz scale window complement evenM evenN rfl).surfaceBody,
      shape.prod ≤ 1048576 := by
  have square : B*B≤1048576 := by
    have sq := Nat.mul_le_mul smallBlock smallBlock
    omega
  have transpose : D*B≤1048576 := by simpa [Nat.mul_comm] using tile
  cases window <;> cases complement <;> cases evenM <;> cases evenN <;>
    simp [Reused.Whole002.TypedSource.widened,ComputeKernel.surfaceBody,statementListShapes,
      statementShapes,algorithmShapes,algorithmListShapes,List.mem_append,smallBlock,tile,
      transpose,square] <;> omega

#axiomsClean quotation_shapes_bounded
end VeriTile.Bench.Repairs.AttentionScoreTileChecked
