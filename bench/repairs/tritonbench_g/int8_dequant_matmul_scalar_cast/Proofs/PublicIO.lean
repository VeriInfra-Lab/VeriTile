import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOImplements
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.PublicArithmetic
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOSafetySource

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.PublicIO
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

theorem programs (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (divisor : F) (schedule : Schedule s c)
    (vals : (IO.programIO q s c hasBias divisor schedule).Inputs) (base : State (Sum F H)) :
    (IO.programIO q s c hasBias divisor schedule).Programs vals base := by
  intro current p _
  let patched:=(IOLayout.memory F H s c hasBias).patchOutputs vals current base
  constructor
  · intro bounds hb
    exact IOSafety.source_safe q s c hasBias ((IO.invocation q s c hasBias divisor p).initial patched)
      (by simpa using p.1.isLt) (by simpa using p.2.isLt) bounds
      (IOSafety.bounds_of_io F H s c hasBias vals bounds hb)
  · obtain ⟨final,ran⟩:=IOInvocation.success q s c hasBias divisor p patched
    exact ⟨final,ran,IOInvocation.frame q s c hasBias divisor p vals patched final ran⟩

theorem generic (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (divisor : F) (schedule : Schedule s c)
    (selected : Selection s c) (plans : q.plans=Quote.selectedPlan s c selected) :
    (IO.programIO q s c hasBias divisor schedule).ImplementsOnCells (IOLayout.domain q.ops s c)
      (IOTrace.result q s c hasBias divisor schedule selected) :=
  IOImplements.assemble q s c hasBias divisor schedule selected plans
    (fun vals base _=>programs q s c hasBias divisor schedule vals base)

theorem real (s : Shape) (c : Config) (selected : Selection s c)
    (hasBias : Bool) (schedule : Schedule s c) :
    (IO.programIO (Quote.idealInteger s c selected) s c hasBias (1/16129) schedule).ImplementsOnCells
      (IOLayout.domain Quote.integerRealOps s c) (IOLayout.real s hasBias) := by
  have result:=generic (Quote.idealInteger s c selected) s c hasBias (1/16129) schedule selected rfl
  rw [PublicArithmetic.real] at result
  exact result

theorem native (profile : FixedISAProfile) (s : Shape) (c : Config)
    (selected : Selection s c) (hasBias : Bool) (schedule : Schedule s c) :
    (IO.programIO (Quote.nativeInteger profile s c selected) s c hasBias Source.runtimeDivisor schedule).ImplementsOnCells
      (IOLayout.domain (fpOps profile) s c) (IOLayout.native profile s c selected hasBias schedule) :=
  generic (Quote.nativeInteger profile s c selected) s c hasBias Source.runtimeDivisor schedule selected rfl

#axiomsClean programs
#axiomsClean generic
#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.PublicIO
