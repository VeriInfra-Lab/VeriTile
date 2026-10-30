import bench.tritonbench_g.fused_recurrent_retention.Proofs.CompilerPlans
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section

/-- Runtime sum plans contain only load indices, zero-initialization, register
moves and scalar adds. They can represent Torch's four thread accumulators,
vectorized head/tail, warp/block/global combines, and TensorIterator splitting.
There is no arithmetic-value argument to plan selection. -/
inductive SumInstruction where
  | zero (dst : Nat)
  | input (dst index : Nat)
  | add (dst left right : Nat)
  | copy (dst src : Nat)
  deriving Repr

def SumInstruction.symbolic (i : SumInstruction) (regs : Nat → SumPlan) : Nat → SumPlan :=
  match i with
  | .zero dst => Function.update regs dst .zero
  | .input dst ix => Function.update regs dst (.input ix)
  | .add dst l r => Function.update regs dst (.add (regs l) (regs r))
  | .copy dst src => Function.update regs dst (regs src)

def SumInstruction.execute (a : Arithmetic α) (x : Nat → α) (i : SumInstruction) (regs : Nat → α) : Nat → α :=
  match i with
  | .zero dst => Function.update regs dst a.zero
  | .input dst ix => Function.update regs dst (x ix)
  | .add dst l r => Function.update regs dst (a.add (regs l) (regs r))
  | .copy dst src => Function.update regs dst (regs src)

theorem sum_instruction_exact (a : Arithmetic α) (x : Nat → α) (i : SumInstruction) (regs : Nat → SumPlan) :
    (fun r => (i.symbolic regs r).eval a x)=i.execute a x (fun r => (regs r).eval a x) := by
  funext r
  cases i <;> simp [SumInstruction.symbolic,SumInstruction.execute,Function.update_apply,SumPlan.eval]
  all_goals split_ifs <;> rfl

def sumProgram (instructions : List SumInstruction) (regs : Nat → SumPlan) : Nat → SumPlan :=
  instructions.foldl (fun r i => i.symbolic r) regs

def sumExecute (a : Arithmetic α) (x : Nat → α) (instructions : List SumInstruction) (regs : Nat → α) : Nat → α :=
  instructions.foldl (fun r i => i.execute a x r) regs

theorem sum_program_exact (a : Arithmetic α) (x : Nat → α) (instructions : List SumInstruction)
    (regs : Nat → SumPlan) :
    (fun r => (sumProgram instructions regs r).eval a x)=
      sumExecute a x instructions (fun r => (regs r).eval a x) := by
  induction instructions generalizing regs with
  | nil => rfl
  | cons i rest ih =>
    change (fun r => (sumProgram rest (i.symbolic regs) r).eval a x)=_
    rw [ih,sum_instruction_exact]
    rfl

structure HostSumPlan (n : Nat) where
  instructions : List SumInstruction
  resultRegister : Nat
  covers : (sumProgram instructions (fun _ => .zero) resultRegister).Covers n

def HostSumPlan.tree (p : HostSumPlan n) : SumPlan :=
  sumProgram p.instructions (fun _ => .zero) p.resultRegister

theorem HostSumPlan.execute_tree (p : HostSumPlan n) (a : Arithmetic α) (x : Nat → α) :
    sumExecute a x p.instructions (fun _ => a.zero) p.resultRegister=p.tree.eval a x := by
  exact (congrFun (sum_program_exact a x p.instructions (fun _ => .zero)) p.resultRegister).symm

/-- A finite compiler dot expression; product/add/FMA arities are bounded by
three even when this complete expression has 32 leaves. -/
structure ScalarDotPlan (n : Nat) where
  expression : DotPlan
  covers : expression.Covers n

/-- Plans are selected from compiler/runtime metadata. Host plans are
selected per output offset and may include different TensorIterator subplans.
Selection depends on runtime metadata and never on tensor numerical values. -/
structure PlanBinding (s : Shape) where
  output : ScalarDotPlan s.BK
  dq : ScalarDotPlan s.BV
  dk : ScalarDotPlan s.BV
  dv : ScalarDotPlan s.BK
  hostO : Nat → HostSumPlan s.NK
  hostQ : Nat → HostSumPlan s.NV
  hostK : Nat → HostSumPlan s.NV
  hostV : Nat → HostSumPlan s.NK

def PlanBinding.plans (b : PlanBinding s) : Plans where
  output := b.output.expression
  dq := b.dq.expression
  dk := b.dk.expression
  dv := b.dv.expression
  sumO := fun i => (b.hostO i).tree
  sumQ := fun i => (b.hostQ i).tree
  sumK := fun i => (b.hostK i).tree
  sumV := fun i => (b.hostV i).tree

theorem PlanBinding.valid (b : PlanBinding s) : b.plans.Valid s :=
  ⟨b.output.covers,b.dq.covers,b.dk.covers,b.dv.covers,
    (fun i => (b.hostO i).covers),(fun i => (b.hostQ i).covers),(fun i => (b.hostK i).covers),(fun i => (b.hostV i).covers)⟩

#axiomsClean sum_instruction_exact
#axiomsClean sum_program_exact
#axiomsClean HostSumPlan.execute_tree
#axiomsClean PlanBinding.valid
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
