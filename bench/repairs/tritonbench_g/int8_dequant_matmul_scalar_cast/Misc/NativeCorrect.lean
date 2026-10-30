import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.FPEquiv
open VeriTile Triton FP.Structural
noncomputable section

/-- Original INT8/INT32/FP32/FP16 source, actual FP32 divisor ABI word and each
output cell's complete split order. Every native output word is observed. -/
abbrev io (profile : FixedISAProfile) (s : Shape) (c : Config)
    (selected : Selection s c) (hasBias : Bool) (schedule : Schedule s c) :=
  IO.programIO (Quote.nativeInteger profile s c selected) s c hasBias Source.runtimeDivisor schedule

/-- Only the original split-K FP16 zero initialization is required. -/
abbrev inputDomain (profile : FixedISAProfile) (s : Shape) (c : Config) :=
  IOLayout.domain (fpOps profile) s c

/-- Exact source-to-native observation under the original fixed instruction
profile. Overflow/saturation, exceptional words, FP16 narrowing and each
address's atomic order are retained. The primitive profile is the model
boundary; no floating rewrite or native-equals-Real claim is introduced. -/
theorem kernel_native (profile : FixedISAProfile) (s : Shape) (c : Config)
    (admitted : Source.Admitted c) (selected : Selection s c)
    (hasBias : Bool) (schedule : Schedule s c) :
    (io profile s c selected hasBias schedule).ImplementsOnCells
      (inputDomain profile s c) (IOLayout.native profile s c selected hasBias schedule) :=
  PublicIO.native profile s c selected hasBias schedule

#stmtSurfaceSubset kernel_native ⊆
  [FixedISAProfile, Shape, Config, Source.Admitted, Selection, Schedule, io, inputDomain,
    Word32, Word16, VeriTile.Bench.TritonBenchG.MatmulDequantInt4.Binary16.instInhabitedValue,
    IOPrograms.Program, IntegerDotPerAddressProgramIO.ImplementsOnCells, IOLayout.native]
#specNonCircular Mathematical.native avoiding [Quote.execute, IntegerDot.exec, gridRun, partialValue]
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.FPEquiv
