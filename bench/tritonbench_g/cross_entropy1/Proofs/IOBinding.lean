import bench.tritonbench_g.cross_entropy1.IO

namespace VeriTile.Bench.Repairs.CrossEntropy1.IOBinding
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

theorem arguments (A : Algebra α) (p : Parameters) (source : ComputeKernel)
    (row block : Nat) (args : RuntimeArgs.Values α) (initial : State α) :
    RuntimeArgs.Bound p ((IO.launch A p source row block args).initial initial) args := by
  constructor <;> simp [IO.launch,Invocation.initial,bindRegisters,State.setReg]

@[simp] theorem row (A : Algebra α) (p : Parameters) (source : ComputeKernel)
    (row block : Nat) (args : RuntimeArgs.Values α) (initial : State α) :
    ((IO.launch A p source row block args).initial initial).pids 0=row := by rfl

@[simp] theorem block (A : Algebra α) (p : Parameters) (source : ComputeKernel)
    (row block : Nat) (args : RuntimeArgs.Values α) (initial : State α) :
    ((IO.launch A p source row block args).initial initial).pids 1=block := by rfl

@[simp] theorem memory (A : Algebra α) (p : Parameters) (source : ComputeKernel)
    (row block : Nat) (args : RuntimeArgs.Values α) (initial : State α) :
    ((IO.launch A p source row block args).initial initial).mem=initial.mem :=
  Invocation.initial_mem _ _

#axiomsClean arguments
end
end VeriTile.Bench.Repairs.CrossEntropy1.IOBinding
