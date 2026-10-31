import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.Mathematical_TypedProgram
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.Mathematical_RealValues
import VeriTile.Triton.Float.Erasure

/-! Complete real execution program obtained by the library's dtype erasure.
The source-body theorem checks every statement, signed loop bound and mask;
the following execution proofs use this exact checked program. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealProgram
open VeriTile.Triton
set_option maxHeartbeats 2400000
set_option maxRecDepth 24000

def numChunks (T BT : Nat) : Nat := (T + BT - 1) / BT

def pointerOp (swapped : Bool) (T BT : Nat) (r : RegionName) : Op .blockPtr [BT] :=
  .makeBlockPtrDynOffsets r
    (if swapped then .mul .nat .nil (.constNat T) (.ref .nat [] "i_bh")
      else .mul .nat .nil (.ref .nat [] "i_bh") (.constNat T))
    [T] [BT] [1] [TypedProgram.offsetOp BT]

def initCode : List Stmt :=
  [.assign .nat [] "i_bh" (.programId 0),
   .assign .real [] "b_z" (.full [] (.const 0))]

theorem constInt_eval (n : Int) (u : BlockState) :
    evalOp (.constInt n) u = some (Tile.scalar n) := by
  simp only [evalOp]

theorem count_eval (T BT : Nat) (u : BlockState) :
    evalOp (TypedProgram.countOp T BT) u = some (Tile.scalar (numChunks T BT)) := by
  simp only [TypedProgram.countOp, evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat,
    Option.bind_some, Option.bind_eq_bind]
  rfl

theorem start_eval (T BT : Nat) (u : BlockState) :
    evalOp (TypedProgram.startOp T BT) u = some (Tile.scalar (1 - (numChunks T BT : Int))) := by
  have hc : evalOp (.castNatToInt (TypedProgram.countOp T BT)) u =
      some (Tile.scalar (numChunks T BT : Int)) := by
    simp only [evalOp, count_eval]
    rfl
  rw [TypedProgram.startOp, evalOp_sub, constInt_eval, evalOp_sub, hc, constInt_eval]
  apply congrArg some
  apply Tile.ext
  rintro ⟨⟩
  change (0 : Int) - ((numChunks T BT : Int) - 1) = 1 - (numChunks T BT : Int)
  omega

#axiomsClean count_eval
#axiomsClean start_eval
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealProgram
