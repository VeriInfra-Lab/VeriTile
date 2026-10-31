import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.DynamicBlockPointerLayout
open VeriTile.Triton FP.Structural
noncomputable section

def signature : CallSignature where
  pointers := [("A",.fp32)]
  scalars := [("sr",.integer),("sc",.integer)]

def body (A : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    p = tl.make_block_ptr(A, (2, 2), (sr, sc), (0, 0), (2, 2), (1, 0))
  }

def kernel (A : RegionName) (sr sc : Nat) : ComputeKernel :=
  (body A).withScalarArguments [ScalarArgument.integer "sr" sr,ScalarArgument.integer "sc" sc]

/-- Runtime ABI values become the actual block-pointer layout. The two
stride expressions remain reads of scalar registers in the source IR. -/
theorem runtime_metadata {α : Type} [Inhabited α] (model : Algebra α)
    (A : RegionName) (sr sc : Nat) (initial : State α) :
    ∃ final, exec model (kernel A sr sc) initial=some final ∧
      final.regs .blockPtr [2,2] "p"=some (fun _=>⟨A,0,[2,2],[2,2],[sr,sc],[0,0]⟩) := by
  refine ⟨?_,?_,?_⟩
  swap
  · simp [kernel,body,ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
      FP.Structural.exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,Region.cast,Option.bind]
    rfl
  · simp

/-- A transposed runtime layout changes the addresses used by the pointer,
not just an unused metadata annotation. -/
theorem runtime_addresses {α : Type} [Inhabited α] (model : Algebra α)
    (A : RegionName) (sr sc : Nat) (initial : State α) :
    ∃ final, exec model (kernel A sr sc) initial=some final ∧
      (final.regs .blockPtr [2,2] "p").map (fun p => (p (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit)).address [1,0])=some sr ∧
      (final.regs .blockPtr [2,2] "p").map (fun p => (p (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit)).address [0,1])=some sc := by
  obtain ⟨final,executed,metadata⟩ := runtime_metadata model A sr sc initial
  refine ⟨final,executed,?_,?_⟩ <;> simp [metadata]

def oneDimensional (A : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers:=[("A",.fp32)],scalars:=[("sr",.integer)]} : CallSignature) in triton {
    p = tl.make_block_ptr(A, (2,), (sr,), (0,), (2,), (0,))
  }

#axiomsClean runtime_metadata
#axiomsClean runtime_addresses
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.DynamicBlockPointerLayout
