import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.NativeIO

/-! Independent ordered native observations of the selected kernels.
Reduction topology, primitive FP32 instructions, FMA contraction and operand
order are the existing compiler model. There is no floating rewrite claim or
new deferred numerical experiment in this task. -/
namespace VeriTile.Bench.Repairs.CrossEntropyOps.FPEquiv
open VeriTile.Triton FP.Structural KernelContract KernelProfile
open scoped VeriTile.Triton.FP.Structural.AddressedProgramIO
noncomputable section

def forwardIO (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (scalars : RuntimeArgs.Values Word) (row block : Nat) : AddressedProgramIO Word Compiler.Term :=
  Interface.Native.forwardIO R I S p scalars row block
def backwardIO (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (scalars : RuntimeArgs.Values Word) (row block : Nat) : AddressedProgramIO Word Compiler.Term :=
  Interface.Native.backwardIO R I S p scalars row block

/-- Logical logits, signed label and runtime scalar words determine the three
ordered native results. No State or source runner appears in this formula. -/
def forwardNative (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (scalars : RuntimeArgs.Values Word) (row block : Nat) :=
  Interface.Native.forwardResult R I S p scalars row block
def backwardNative (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (scalars : RuntimeArgs.Values Word) (row block : Nat) :=
  Interface.Native.backwardResult R I S p scalars row block

theorem native_forward (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (scalars : RuntimeArgs.Values Word) (row block : Nat)
    (positive : 0 < p.block) (separate : p.lse ≠ p.logits) (outputs : p.lse ≠ p.loss)
    (zLse : p.zloss ≠ p.lse) (zLoss : p.zloss ≠ p.loss) :
    forwardIO R I S p scalars row block ⊨ forwardNative R I S p scalars row block :=
  Interface.Native.forward_implements R I S p scalars row block positive separate outputs zLse zLoss

theorem native_backward (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (scalars : RuntimeArgs.Values Word) (row block : Nat) :
    backwardIO R I S p scalars row block ⊨ backwardNative R I S p scalars row block :=
  Interface.Native.backward_implements R I S p scalars row block

#axiomsClean native_forward
#axiomsClean native_backward
#specNonCircular forwardNative avoiding [forwardSyntax, backwardSyntax, exec, run]
#specNonCircular backwardNative avoiding [forwardSyntax, backwardSyntax, exec, run]
#stmtSurfaceSubset native_forward ⊆
  [AddressedProgramIO.Implements, RegionName, Compiler.Term, SoftmaxFlaggems.Native.instInhabitedTerm,
   forwardIO, forwardNative, Compiler.Reductions, Compiler.Instructions,
   Compiler.Schedule, Parameters, Parameters.toParameters, Parameters.zloss, RuntimeArgs.Values, Word,
   CrossEntropy1.Parameters.block, CrossEntropy1.Parameters.lse, CrossEntropy1.Parameters.logits,
   CrossEntropy1.Parameters.loss]
#stmtSurfaceSubset native_backward ⊆
  [AddressedProgramIO.Implements, Compiler.Term, SoftmaxFlaggems.Native.instInhabitedTerm,
   backwardIO, backwardNative, Compiler.Reductions, Compiler.Instructions,
   Compiler.Schedule, Parameters, RuntimeArgs.Values, Word]
end
end VeriTile.Bench.Repairs.CrossEntropyOps.FPEquiv
