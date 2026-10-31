import VeriTile.Triton.DSL
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.Mathematical_TypedContext

/-! Checked decomposition of the complete typed source. The loop's genuine
signed bounds, generated index binding and complete floating suffix remain
present. These equalities also check the candidate's exact rewrite site. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedProgram
open VeriTile.Triton
set_option maxHeartbeats 1800000
set_option maxRecDepth 18000

def countOp (T BT : Nat) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat T) (.constNat BT))
    (.constNat 1)) (.constNat BT)

def startOp (T BT : Nat) : Op .int [] :=
  .sub .int .nil (.constInt 0) (.sub .int .nil (.castNatToInt (countOp T BT)) (.constInt 1))

def offsetOp (BT : Nat) : Op .nat [] :=
  .castIntToNat (.mul .int .nil (.ref .int [] "i_t") (.constInt (BT : Int)))

end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedProgram
