import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.PublicResult

namespace VeriTile.Bench.Repairs.Conv2d
open VeriTile Triton FP.Structural
noncomputable section

/-- Complete source-to-native IO for all original flags and raw FP32 words.
The explicit local laws concern the original scalar/conversion/fixed-MMA
primitive model; this is not a floating transformation or a second kernel. -/
theorem fp_equivalent (s : Shape) (valid : Valid s) (flags : Flags)
    (actual numerics : KernelNative.Instructions)
    (laws : KernelNative.LocalLaws actual (KernelNative.software numerics)) :
    (ConvIO.nativeIO actual s flags).Implements (ConvIO.nativeTarget numerics s flags) :=
  ConvIO.native_implements s valid flags actual numerics laws

#stmtSurfaceSubset fp_equivalent ⊆
  [Shape,Valid,Flags,Flags.mk,KernelNative.Instructions,KernelNative.LocalLaws,KernelNative.software,
   Native.Word,Native.Byte,ConvIO.Payload,KernelNumeric.Value,KernelNumeric.instInhabitedValue,
   ConvIO.nativeIO,ConvIO.nativeTarget,ProgramIO.Implements]
#specNonCircular ConvIO.nativeTarget avoiding
  [KernelSource.kernel,KernelGrid.launch,FP.Structural.exec,KernelGrid.value]
#axiomsClean fp_equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Conv2d
