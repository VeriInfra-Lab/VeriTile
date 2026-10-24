import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.PublicResult

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton Floating
noncomputable section
/-- Complete persistent launch, two strided matrix inputs and one output.
Storage is the selected FP16 or FP8 profile; fused accumulation remains FP32.
There is no private scratch. Empty matrices retain the original empty grid. -/
abbrev nativeMatmulIO (p : Core.Parameters) (plan : DotLayout p.problem.profile.blockK) (native : Floating.Numerics) : ProgramIO Floating.Word Floating.Word :=
  PublicData.io (Native.arithmetic p.problem.profile plan native) p

/-- Independent native target: four ordered ISA stages per K block, actual
incoming accumulators, masked padding and the selected output conversion. -/
def nativeExpected (p : Core.Parameters) (plan : DotLayout p.problem.profile.blockK)
    (reference : Native.Reference) :=
  PublicData.result (Native.arithmetic p.problem.profile plan reference.numerics) p

/-- Source-to-native observation for every input word. Local scalar-conversion
and named MMA ports retain their existing scope; they are primitive-model
boundaries, not an algebraic rewrite or a native-equals-Real assertion. -/
theorem kernel_fp_equiv (p : Core.Parameters) (b : RawKernel.Buffers)
    (layout : RawKernel.Layout p b) (sms : 0<p.problem.SMS)
    (plan : DotLayout p.problem.profile.blockK) (native : Floating.Numerics) (reference : Native.Reference)
    (ports : Floating.LocalPorts native reference.numerics Native.allWords) :
    (nativeMatmulIO p plan native).Implements (nativeExpected p plan reference) :=
  PublicData.native_correct p b layout sms plan native reference ports

#axiomsClean kernel_fp_equiv
#specNonCircular nativeExpected avoiding [Core.executeGrid,Core.program,Source.run,Source.accumulator]
#specNonCircular Floating.LocalPorts avoiding [Core.executeGrid,Targets.result,Source.run,Source.accumulator]
#stmtSurfaceSubset kernel_fp_equiv ⊆
  [Core.Parameters,Core.Parameters.problem,Problem.SMS,Problem.profile,Storage.blockK,
   RawKernel.Buffers,RawKernel.Layout,DotLayout,Floating.Numerics,Native.Reference,
   Native.Reference.numerics,Floating.LocalPorts,Native.allWords,Floating.Word,
   nativeMatmulIO,nativeExpected,ProgramIO.Implements]

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast
