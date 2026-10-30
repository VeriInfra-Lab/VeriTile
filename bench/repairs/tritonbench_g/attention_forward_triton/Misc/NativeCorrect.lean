import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.PublicResult

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.FPEquiv
open VeriTile Triton FP.Structural Source Whole
noncomputable section

/-- Complete source-to-native observation for all FP32 input words and every
legal grid order. Fixed K8 TF32 QK and K16 FP16 PV ports, original reduction
trees, cast locations and replica ownership remain explicit. This source has
no optimization claim or floating transformation assumption. -/
theorem attention_kernel_native_raw (instructions : Native.Instructions)
    (p : Parameters) (sep : Separated p) (injective : RawKernel.OutputInjective p)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    (AttentionIO.nativeIO instructions p work).Implements (AttentionIO.nativeTarget instructions p) :=
  AttentionIO.native_implements instructions p sep injective work legal

#stmtSurfaceSubset attention_kernel_native_raw ⊆
  [Native.Word,Native.Expr,Native.instInhabitedExpr,Native.Instructions,Parameters,Separated,RawKernel.OutputInjective,Program,programs,grid,
   AttentionIO.nativeIO,AttentionIO.nativeTarget,ProgramIO.Implements]
#specNonCircular AttentionIO.nativeTarget avoiding
  [Source.kernel,runPrograms,FP.Structural.exec,Source.accumulated,Source.update,Whole.answer]
#axiomsClean attention_kernel_native_raw
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.FPEquiv
