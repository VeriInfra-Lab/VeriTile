import bench.repairs.tritonbench_g.cross_entropy1.Proofs.PublicIO
import bench.repairs.tritonbench_g.cross_entropy1.Misc.RawNative
import VeriTile.Triton.Float.ObservedProgramIO

namespace VeriTile.Bench.Repairs.CrossEntropy1.FPEquiv
open VeriTile.Triton FP.Structural KernelProfile
noncomputable section

/-- Actual runtime scalar terms, integer labels and unchanged FP32 source.
Inputs retain arbitrary symbolic terms, including all original native words. -/
def forwardIO (R : Compiler.Reductions) (p : Parameters) (row block : Nat)
    (args : RuntimeArgs.Values Compiler.Term) : ProgramIO Compiler.Term Compiler.Term :=
  IO.forward (Compiler.algebra R) p row block args id id

def backwardIO (R : Compiler.Reductions) (p : Parameters) (row block : Nat)
    (args : RuntimeArgs.Values Compiler.Term) : ProgramIO Compiler.Term Compiler.Term :=
  IO.backward (Compiler.algebra R) p row block args id id

def forwardObserver (p : Parameters) (I : Compiler.Instructions) (S : Compiler.Schedule) :
    (o : Fin (IO.forwardOutputs p).length) → TileIndex (IO.forwardOutputs p)[o].shape →
      Compiler.Term → BitVec 32 := fun _ _ => Compiler.eval I S

def backwardObserver (p : Parameters) (I : Compiler.Instructions) (S : Compiler.Schedule) :
    (o : Fin 1) → TileIndex (([⟨.real,[p.block],p.gradient⟩] : List IOChannel)[o]).shape →
      Compiler.Term → BitVec 32 := fun _ _ => Compiler.eval I S

/-- Independent source-ordered expressions are observed under the original
FP32 instruction model and structural compiler schedule, including contraction.
No floating transformation axiom or native-to-Real equality is asserted. -/
def forwardTarget (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (block : Nat) (args : RuntimeArgs.Values Compiler.Term)
    (vals : ChannelValues Compiler.Term (IO.forwardInputs p)) :
    ChannelValues (BitVec 32) (IO.forwardOutputs p) :=
  let result := Reference.forward (Compiler.algebra R) p args block
    (IO.vector (vals (0 : Fin 2))) (vals (1 : Fin 2) PUnit.unit)
  Fin.cases (fun _ => Compiler.eval I S result.1)
    (fun next => Fin.cases (fun _ => Compiler.eval I S result.2)
      (fun impossible => Fin.elim0 impossible) next)

def backwardTarget (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (block : Nat) (args : RuntimeArgs.Values Compiler.Term)
    (vals : ChannelValues Compiler.Term (IO.backwardInputs p)) :
    ChannelValues (BitVec 32) [⟨.real,[p.block],p.gradient⟩] :=
  Fin.cases (fun j => Compiler.eval I S (Reference.backward (Compiler.algebra R) p args block
    (IO.vector (vals (0 : Fin 4))) (vals (1 : Fin 4) PUnit.unit)
    (vals (2 : Fin 4) PUnit.unit) (vals (3 : Fin 4) PUnit.unit) j.1))
    (fun impossible => Fin.elim0 impossible)

theorem forward_kernel (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (row block : Nat) (args : RuntimeArgs.Values Compiler.Term)
    (positive : 0<p.block) (separate : p.lse≠p.logits)
    (outputs : p.lse≠p.loss) :
    (forwardIO R p row block args).Observes (forwardObserver p I S)
      (forwardTarget R I S p block args) := by
  have result := ProgramIO.Observes.of_implements
    (PublicIO.forward (Compiler.algebra R) p row block args positive separate outputs)
    (fun _ _ => Compiler.eval I S)
  convert result using 1
  funext vals port lane
  fin_cases port <;> rfl

theorem backward_kernel (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (row block : Nat) (args : RuntimeArgs.Values Compiler.Term) :
    (backwardIO R p row block args).Observes (backwardObserver p I S)
      (backwardTarget R I S p block args) := by
  have result := ProgramIO.Observes.of_implements
    (PublicIO.backward (Compiler.algebra R) p row block args) (fun _ _ => Compiler.eval I S)
  convert result using 1
  funext vals port lane
  fin_cases port
  rfl

export KernelProfile (forward_square_fused backward_correction_fused)

#stmtSurfaceSubset forward_kernel ⊆
  [Compiler.Reductions,Compiler.Instructions,Compiler.Schedule,Parameters,Parameters.block,
   Parameters.lse,Parameters.logits,Parameters.loss,RegionName,RuntimeArgs.Values,Compiler.Term,
   ProgramIO.Observes,forwardIO,forwardObserver,forwardTarget,
   VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.instInhabitedTerm]
#stmtSurfaceSubset backward_kernel ⊆
  [Compiler.Reductions,Compiler.Instructions,Compiler.Schedule,Parameters,RuntimeArgs.Values,
   Compiler.Term,ProgramIO.Observes,backwardIO,backwardObserver,backwardTarget,
   VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.instInhabitedTerm]
#axiomsClean forward_kernel
#axiomsClean backward_kernel
#specNonCircular forwardTarget avoiding [forwardSyntax,backwardSyntax,FP.Structural.exec]
#specNonCircular backwardTarget avoiding [forwardSyntax,backwardSyntax,FP.Structural.exec]
-- The exact original word-seeded entry and scalar ABI statements remain in
-- Proofs/RawNative.lean; this generic observation includes those source terms.
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.FPEquiv
