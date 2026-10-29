import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.Source
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.TypedContext

/-! Checked decomposition of the complete typed source. The loop's genuine
signed bounds, generated index binding and complete floating suffix remain
present. These equalities also check the candidate's exact rewrite site. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedProgram
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1800000
set_option maxRecDepth 18000

def countOp (T BT : Nat) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat T) (.constNat BT))
    (.constNat 1)) (.constNat BT)

def startOp (T BT : Nat) : Op .int [] :=
  .sub .int .nil (.constInt 0) (.sub .int .nil (.castNatToInt (countOp T BT)) (.constInt 1))

def offsetOp (BT : Nat) : Op .nat [] :=
  .castIntToNat (.mul .int .nil (.ref .int [] "i_t") (.constInt (BT : Int)))

def indexStmt : ComputeStmt :=
  .assign .int [] "i_t" (.alg (.sub .int .nil (.constInt 0) (.ref .int [] "__rev_i_t")))

def parts (swapped : Bool) (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) : Source.IntRange :=
  (Source.intRange? (source swapped dtype out_dtype s o T BT).surfaceBody "__rev_i_t").getD
    ⟨[], .constInt 0, .constInt 0, .constNat 0, [], []⟩

theorem source_split (swapped : Bool) (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) :
    (source swapped dtype out_dtype s o T BT).surfaceBody =
      (source Bool.false dtype out_dtype s o T BT).surfaceBody.take 2 ++
      [.forRangeInt "__rev_i_t" (startOp T BT) (.constInt 1) (.constNat 1)
        (parts swapped dtype out_dtype s o T BT).body] := by
  cases swapped <;> rfl

theorem body_split (swapped : Bool) (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) :
    (parts swapped dtype out_dtype s o T BT).body = [indexStmt] ++
      TypedContext.pointerStmt swapped s T BT (offsetOp BT) ::
      (parts Bool.false dtype out_dtype s o T BT).body.drop 2 := by
  cases swapped <;> rfl

theorem body_statement_count (swapped : Bool) (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) :
    (parts swapped dtype out_dtype s o T BT).body.length = 8 := by
  cases swapped <;> rfl

#axiomsClean source_split
#axiomsClean body_split
#axiomsClean body_statement_count
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedProgram
