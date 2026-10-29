import bench.repairs.tritonbench_g.cross_entropy2.Proofs.PublicIO
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Compiler
import VeriTile.Triton.Float.ObservedProgramIO

namespace VeriTile.Bench.Repairs.CrossEntropy2.FPEquiv
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
    (fun next => Fin.cases (fun _ => Compiler.eval I S result.2.1)
      (fun next => Fin.cases (fun _ => Compiler.eval I S result.2.2)
        (fun impossible => Fin.elim0 impossible) next) next)

def backwardTarget (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (block : Nat) (args : RuntimeArgs.Values Compiler.Term)
    (vals : ChannelValues Compiler.Term (IO.backwardInputs p)) :
    ChannelValues (BitVec 32) [⟨.real,[p.block],p.gradient⟩] :=
  Fin.cases (fun j => Compiler.eval I S (Reference.backward (Compiler.algebra R) p args block
    (IO.vector (vals (0 : Fin 4))) (vals (1 : Fin 4) PUnit.unit)
    (vals (2 : Fin 4) PUnit.unit) (vals (3 : Fin 4) PUnit.unit) j.1))
    (fun impossible => Fin.elim0 impossible)

theorem forward (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (row block : Nat) (args : RuntimeArgs.Values Compiler.Term)
    (positive : 0<p.block) (separate : p.lse≠p.logits)
    (outputs : p.lse≠p.loss ∧ p.lse≠p.zLoss ∧ p.loss≠p.zLoss) :
    (forwardIO R p row block args).Observes (fun _ _ => Compiler.eval I S)
      (forwardTarget R I S p block args) := by
  have result := ProgramIO.Observes.of_implements
    (PublicIO.forward (Compiler.algebra R) p row block args positive separate outputs)
    (fun _ _ => Compiler.eval I S)
  convert result using 1
  funext vals port lane
  fin_cases port <;> rfl

theorem backward (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (row block : Nat) (args : RuntimeArgs.Values Compiler.Term) :
    (backwardIO R p row block args).Observes (fun _ _ => Compiler.eval I S)
      (backwardTarget R I S p block args) := by
  have result := ProgramIO.Observes.of_implements
    (PublicIO.backward (Compiler.algebra R) p row block args) (fun _ _ => Compiler.eval I S)
  convert result using 1
  funext vals port lane
  fin_cases port
  rfl

#axiomsClean forward
#axiomsClean backward
#specNonCircular forwardTarget avoiding [forwardSyntax,backwardSyntax,FP.Structural.exec]
#specNonCircular backwardTarget avoiding [forwardSyntax,backwardSyntax,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.CrossEntropy2.FPEquiv
