import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.NativeImplementation

/-! Retained single-source native correctness. These ordinary theorems keep
the original names, full contracts and proofs; they do not assert an
implementation-pair floating-point equivalence. -/

/-! Complete native source observation. NativeIO exposes the original FP32
quotation, six named arrays, finite access windows, raw scale ABI and padded
M/L/O outputs. Raw.LocalLaws compares the same nine primitive/ISA ports; it is
not a floating transformation assumption or an equality between native and Real.
Full raw-byte reachability, publication and Tensor statements remain in Auxiliary.
-/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host
noncomputable section

theorem fp_equiv (c : Call) (_valid : Valid c) (shape : Native.Shape c)
    (schedule : Raw.Schedule c) (actual reference : Raw.Ops)
    (laws : Raw.LocalLaws actual reference) (scale : Native.Word) :
    (NativeInterface.programIO c).Observes
      (NativeInterface.observer c shape schedule actual scale)
      (NativeInterface.formula c shape schedule reference scale) :=
  NativeInterface.observes c shape schedule actual reference laws scale

#axiomsClean fp_equiv
#stmtSurfaceSubset fp_equiv ⊆
  [Call,Valid,Native.Shape,Raw.Schedule,Raw.Ops,Raw.LocalLaws,Native.Word,
   Native.Expr,Native.instInhabitedExpr,ArrayProgramIO.Observes,
   NativeInterface.programIO,NativeInterface.observer,NativeInterface.formula]
#specNonCircular NativeInterface.formula avoiding
  [Kernels.source,FP.Structural.exec,FP.Structural.run,runSchedule,Reference.complete_grid]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast
