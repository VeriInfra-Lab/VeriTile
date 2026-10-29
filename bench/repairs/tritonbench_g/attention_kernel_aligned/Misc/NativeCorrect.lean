import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.IOSafety
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16IOSafety

/-! Native observation of both complete original sources. Arbitrary symbolic
input payloads are retained; the output observer uses the original physical
owner. Instruction primitives and compiler routing remain explicit boundaries.
No floating transformation, new numerical law, or native-to-Real equality is
asserted. The BF16 plan retains its nativeSerial=true route. -/
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.FPEquiv
open VeriTile Triton FP.Structural Full Whole Interface
noncomputable section

theorem kernel_native (I : Native.Instructions) (p : Parameters) (g : Guard p)
    (separate : Whole.Nonempty p → Separated p) (plan : RawKernel.CompilerPlan p)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    (nativeIO p work).Observes (nativeObserver I p work) (nativeFormula I p g.selected work) :=
  native_observes I p g separate plan work order

#axiomsClean kernel_native
#stmtSurfaceSubset kernel_native ⊆
  [Native.Instructions,Parameters,Guard,Guard.selected,Whole.Nonempty,Separated,RawKernel.CompilerPlan,
   Program,programs,grid,nativeIO,nativeObserver,nativeFormula,ProgramIO.Observes,
   Native.Expr,Native.instInhabitedExpr]
#specNonCircular KernelOnly.Specification.outputValue avoiding
  [Source.source,FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.rounds,Source.advance]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.FPEquiv

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.FPEquiv
open VeriTile Triton FP.Structural Full Whole Interface
noncomputable section

theorem kernel_native (I : Native.Instructions) (p : Parameters) (g : Guard p)
    (separate : Whole.Nonempty p → Separated p) (plan : RawKernel.CompilerPlan p)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    (nativeIO p work).Observes (nativeObserver I p work) (nativeFormula I p g.selected work) :=
  native_observes I p g separate plan work order

#axiomsClean kernel_native
#stmtSurfaceSubset kernel_native ⊆
  [Native.Instructions,Parameters,Guard,
   VeriTile.Bench.Repairs.AttentionKernelAligned.Whole.Guard.selected,
   Whole.Nonempty,Separated,RawKernel.CompilerPlan,
   Program,programs,grid,nativeIO,nativeObserver,nativeFormula,ProgramIO.Observes,
   Native.Expr,Native.instInhabitedExpr]
#specNonCircular KernelOnly.Specification.outputValue avoiding
  [Source.source,FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.rounds,Source.advance]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.FPEquiv
