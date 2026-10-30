/- Auxiliary single-source native correctness. No program-pair equivalence is claimed. -/
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativeInterface

namespace VeriTile.Bench.Repairs.LayerNormLiger.Equivalence
open VeriTile.Triton FP.Structural
noncomputable section
set_option autoImplicit false

def forwardIO := LayerIO.nativeForwardIO
def backwardIO := LayerIO.nativeBackwardIO
def forwardReference := LayerIO.forwardReference
def backwardReference := LayerIO.backwardReference
def forwardNative := LayerIO.forwardNative
def backwardNative := LayerIO.backwardNative

/-- Exact independent ordered FP32 expression graph and its native primitive
observation. Mean/RSTD/Y aliases and arbitrary strides keep source write order.
Primitives and contraction/tree choices are model parameters. This auxiliary
result introduces no experimental transformation assumption. -/
theorem layer_norm_forward_kernel_fp32 (P : NativePlan.Primitives)
    (contraction : NativePlan.Contraction) (trees : NativePlan.Trees)
    (p : Forward.Parameters) (pid : Nat) :
    (forwardIO contraction trees p pid).Implements (forwardReference contraction trees p pid) ∧
    (forwardIO contraction trees p pid).Observes (fun _ _ => NativePlan.Expr.eval P)
      (forwardNative P contraction trees p pid) :=
  LayerIO.forward_native P contraction trees p pid

/-- Complete row-loop native graph, DX and both partial gradients, including
read-after-write under arbitrary region/stride aliases. No mathematical
separation premise, finite-input restriction or native=Real law is added. -/
theorem layer_norm_backward_kernel_fp32 (P : NativePlan.Primitives)
    (contraction : NativePlan.Contraction) (trees : NativePlan.Trees)
    (p : Backward.Parameters) (pid : Nat) :
    (backwardIO contraction trees p pid).Implements (backwardReference contraction trees p pid) ∧
    (backwardIO contraction trees p pid).Observes (fun _ _ => NativePlan.Expr.eval P)
      (backwardNative P contraction trees p pid) :=
  LayerIO.backward_native P contraction trees p pid

#stmtSurfaceSubset layer_norm_forward_kernel_fp32 ⊆
  [forwardIO,forwardReference,forwardNative,ProgramIO.Implements,ProgramIO.Observes,
   NativePlan.Primitives,NativePlan.Contraction,NativePlan.Trees,NativePlan.Expr.eval,
   NativePlan.Expr,NativePlan.instInhabitedExpr,Forward.Parameters,FP.Value,FP.Format.fp32,
   IOChannel,MemoryIO.outputs,ProgramIO.toMemoryIO,TileIndex,IOChannel.shape,ProgramIO.mapOutputs._proof_1]
#stmtSurfaceSubset layer_norm_backward_kernel_fp32 ⊆
  [backwardIO,backwardReference,backwardNative,ProgramIO.Implements,ProgramIO.Observes,
   NativePlan.Primitives,NativePlan.Contraction,NativePlan.Trees,NativePlan.Expr.eval,
   NativePlan.Expr,NativePlan.instInhabitedExpr,Backward.Parameters,FP.Value,FP.Format.fp32,
   IOChannel,MemoryIO.outputs,ProgramIO.toMemoryIO,TileIndex,IOChannel.shape,ProgramIO.mapOutputs._proof_1]
#specNonCircular forwardReference avoiding [Forward.kernel,FP.Structural.exec,Forward.reference]
#specNonCircular backwardReference avoiding [Backward.kernel,FP.Structural.exec,Backward.reference]
#axiomsClean layer_norm_forward_kernel_fp32
#axiomsClean layer_norm_backward_kernel_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormLiger.Equivalence
