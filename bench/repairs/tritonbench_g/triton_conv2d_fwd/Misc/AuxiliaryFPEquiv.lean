import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.PublicResult

/-! Auxiliary specialization of the original convolution kernel at its
production-default flags. The historical name and complete statement remain. -/
namespace VeriTile.Bench.Repairs.Conv2d
open VeriTile Triton FP.Structural
noncomputable section

/-- The production-default branch observes the IEEE RN-even FP32 FMA circuit.
Only the existing scalar FMA law is needed; all padded operations remain. -/
theorem fp_default (s : Shape) (valid : Valid s) (flags : Flags)
    (defaults : flags=⟨Bool.false,Bool.false⟩) (actual : KernelNative.Instructions)
    (fmaLaw : ∀ a b c,actual.fmaRN a b c=Software.fmaRN a b c) :
    (ConvIO.nativeIO actual s flags).Implements (ConvIO.defaultTarget actual s flags) :=
  ConvIO.default_implements s valid flags defaults actual fmaLaw

#stmtSurfaceSubset fp_default ⊆
  [Shape,Valid,Flags,Flags.mk,KernelNative.Instructions,KernelNative.Instructions.fmaRN,Software.fmaRN,
   Native.Word,Native.Byte,ConvIO.Payload,KernelNumeric.Value,KernelNumeric.instInhabitedValue,
   ConvIO.nativeIO,ConvIO.defaultTarget,ProgramIO.Implements]
#specNonCircular ConvIO.defaultTarget avoiding
  [KernelSource.kernel,KernelGrid.launch,FP.Structural.exec,KernelGrid.value]
#axiomsClean fp_default
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Conv2d
