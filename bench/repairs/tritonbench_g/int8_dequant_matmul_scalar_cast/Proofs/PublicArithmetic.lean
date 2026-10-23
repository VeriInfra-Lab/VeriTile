import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOTrace
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.FiniteRealResult

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.PublicArithmetic
open VeriTile Triton FP.Structural
noncomputable section

theorem real (s : Shape) (c : Config) (hasBias : Bool) (schedule : Schedule s c)
    (selected : Selection s c) :
    IOTrace.result (Quote.idealInteger s c selected) s c hasBias (1/16129) schedule selected=
      IOLayout.real s hasBias := by
  funext vals o j
  have ho : o=0 := Fin.eq_zero o
  subst o
  obtain ⟨r,n,u⟩:=j
  cases u
  exact FiniteRealResult.real s c selected hasBias schedule (IOLayout.data Quote.integerRealOps s vals) (r,n)

theorem native (profile : FixedISAProfile) (s : Shape) (c : Config) (hasBias : Bool)
    (schedule : Schedule s c) (selected : Selection s c) :
    IOTrace.result (Quote.nativeInteger profile s c selected) s c hasBias Source.runtimeDivisor schedule selected=
      IOLayout.native profile s c selected hasBias schedule := rfl

#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.PublicArithmetic
