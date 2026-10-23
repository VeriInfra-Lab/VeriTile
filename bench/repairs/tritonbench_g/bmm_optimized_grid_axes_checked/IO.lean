import VeriTile.Triton.Float.ProgramEquivalence
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.SourceGraph
import bench.repairs.tritonbench_g.bmm_optimized.Proofs.Native
import VeriTile.Triton.Float.LogicalProgramIO

/-! Public contiguous row-major BMM channels. Each finite channel is flattened
using its displayed capacity; `Shape` retains batch/M/N/K. The selected source
fixes FP32 loads, output storage, 32-wide contraction and allow_tf32=False.
The graph carrier records that source's ordered fused accumulator interpretation;
the logical target is an independent array formula. -/
namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Interface
open VeriTile.Triton FP.Structural BmmOptimized
noncomputable section

abbrev inputs (s : Shape) : List IOChannel :=
  [⟨.real,[s.batch*s.m*s.k],"A"⟩,⟨.real,[s.batch*s.k*s.n],"B"⟩]
abbrev outputs (s : Shape) : List IOChannel := [⟨.real,[s.size],"O"⟩]

/-- Exact original three-axis launch, including per-CTA fresh registers. -/
def invocation (s : Shape) (c : Config) (p : Scalar α) (rest : Algebra α)
    (pid : Program) : Invocation (SourceGraph.Node α) where
  kernel := Source.kernel (Source.parameters s c)
  backend := SourceGraph.engine p rest
  clearRegisters := Bool.true
  pids := fun ax=>if ax=0 then pid.1 else if ax=1 then pid.2.1 else if ax=2 then pid.2.2 else 0
  numPids := some (fun ax=>if ax=0 then s.gx c else if ax=1 then s.gy c else if ax=2 then s.batch else 1)

def io (s : Shape) (c : Config) (p : Scalar α) (rest : Algebra α)
    (order : List Program) : ProgramIO α (SourceGraph.Node α) where
  inputs := inputs s
  outputs := outputs s
  buffers := ["A","B","O"]
  scratch := []
  read := fun i _=>by
    refine Fin.cases (fun j=>j.1.val) ?_ i
    intro k
    have hk : k=0 := Fin.eq_zero k
    subst k
    exact fun j=>j.1.val
  inputMask := fun _ _ _=>True
  write := fun o _=>by
    have ho : o=0 := Fin.eq_zero o
    subst o
    exact fun j=>j.1.val
  outputMask := fun _ _ _=>True
  scratchWrite := fun k=>k.elim0
  scratchMask := fun k=>k.elim0
  schedule := fun _=>order.map (invocation s c p rest)
  encode := SourceGraph.Node.value
  observe := fun _ _=>SourceGraph.observe p

/-- Pure finite-array extension, used only to state the independent formula.
No source memory, execution state or source-value callback occurs here. -/
def arrays (zero : α) (s : Shape) (vals : ChannelValues α (inputs s)) : Memory α
  | .a,i=>if h : i<s.batch*s.m*s.k then vals 0 (⟨i,h⟩,PUnit.unit) else zero
  | .b,i=>if h : i<s.batch*s.k*s.n then vals 1 (⟨i,h⟩,PUnit.unit) else zero
  | _,_=>zero

def outputCell (s : Shape) (i : Fin s.size) : BmmOptimized.Cell :=
  ⟨i.val/(s.m*s.n),(i.val/s.n)%s.m,i.val%s.n⟩

/-- Batch matrix multiplication over mathematical Real inputs. -/
def realFormula (s : Shape) (vals : ChannelValues ℝ (inputs s)) :
    ChannelValues ℝ (outputs s) := by
  intro o
  have ho : o=0 := Fin.eq_zero o
  subst o
  exact fun j=>mathematical s (arrays 0 s vals) (outputCell s j.1)

/-- Independent ordered scalar chain, including final padded zero FMAs.
Every FP32 word is admitted; no finite-word or reassociation premise is added. -/
def scalarFormula (s : Shape) (p : Scalar α) (vals : ChannelValues α (inputs s)) :
    ChannelValues α (outputs s) := by
  intro o
  have ho : o=0 := Fin.eq_zero o
  subst o
  exact fun j=>ordered s p (arrays p.zero s vals) (outputCell s j.1)

def realIO (s : Shape) (c : Config) (rest : Algebra ℝ) (order : List Program) :=
  io s c realScalar rest order
def nativeIO (s : Shape) (c : Config) (ops : Native.Instructions)
    (rest : Algebra Native.Word) (order : List Program) :=
  io s c (Native.scalar ops) rest order

end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Interface

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.PairIO
open VeriTile Triton FP.Structural BmmOptimized Interface
noncomputable section
variable {A : Type} [Inhabited A]
def invocation (s : Shape) (c : Config) (p : Scalar A) (rest : Algebra A)
    (pid : Program) : Invocation (SourceGraph.Node A) :=
  { Interface.invocation s c p rest pid with
    kernel := Source.quotientRemainder (Source.parameters s c) }

def candidate (s : Shape) (c : Config) (p : Scalar A) (rest : Algebra A)
    (ps : List Program) : ProgramIO.Variant (Interface.io s c p rest ps) where
  schedule := fun _ => ps.map (invocation s c p rest)
  observe := fun _ _ => SourceGraph.observe p

end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.PairIO

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Interface
open VeriTile Triton FP.Structural BmmOptimized
noncomputable section

def tripleIndex (a b c : Nat) : ((Fin a × Fin b) × Fin c) ≃ Fin (a*b*c) :=
  (Equiv.prodCongr finProdFinEquiv (Equiv.refl _)).trans finProdFinEquiv

def triple (a b c : Nat) : (Fin (a*b*c) → ℝ) ≃ (Fin a → Fin b → Fin c → ℝ) where
  toFun := fun f i j k => f (tripleIndex a b c ((i,j),k))
  invFun := fun f index =>
    let cell := (tripleIndex a b c).symm index
    f cell.1.1 cell.1.2 cell.2
  left_inv := by
    intro f
    funext index
    change f (tripleIndex a b c ((tripleIndex a b c).symm index)) = f index
    rw [Equiv.apply_symm_apply]
  right_inv := by
    intro f
    funext i j k
    change f ((tripleIndex a b c).symm (tripleIndex a b c ((i,j),k))).1.1
      ((tripleIndex a b c).symm (tripleIndex a b c ((i,j),k))).1.2
      ((tripleIndex a b c).symm (tripleIndex a b c ((i,j),k))).2 = f i j k
    rw [Equiv.symm_apply_apply]

structure LogicalInputs (s : Shape) where
  a : Fin s.batch → Fin s.m → Fin s.k → ℝ
  b : Fin s.batch → Fin s.k → Fin s.n → ℝ

def logicalInputsEquiv (s : Shape) : ChannelValues ℝ (inputs s) ≃ LogicalInputs s where
  toFun := fun values => ⟨triple s.batch s.m s.k (fun i => values 0 (i,())),
    triple s.batch s.k s.n (fun i => values 1 (i,()))⟩
  invFun := fun xs => Fin.cases (fun i => (triple s.batch s.m s.k).symm xs.a i.1)
    (Fin.cases (fun i => (triple s.batch s.k s.n).symm xs.b i.1) (fun i => i.elim0))
  left_inv := by
    intro values
    funext channel j
    fin_cases channel
    · rcases j with ⟨j,⟨⟩⟩
      exact congrFun ((triple s.batch s.m s.k).symm_apply_apply (fun i => values 0 (i,()))) j
    · rcases j with ⟨j,⟨⟩⟩
      exact congrFun ((triple s.batch s.k s.n).symm_apply_apply (fun i => values 1 (i,()))) j
  right_inv := by
    intro xs
    change LogicalInputs.mk (triple s.batch s.m s.k ((triple s.batch s.m s.k).symm xs.a))
      (triple s.batch s.k s.n ((triple s.batch s.k s.n).symm xs.b)) = xs
    rw [Equiv.apply_symm_apply,Equiv.apply_symm_apply]

def logicalOutputsEquiv (s : Shape) : ChannelValues ℝ (outputs s) ≃
    (Fin s.batch → Fin s.m → Fin s.n → ℝ) :=
  (ChannelValues.singleEquiv ℝ ⟨.real,[s.size],"O"⟩).trans
    ((Values.vectorEquiv ℝ .real s.size).trans (triple s.batch s.m s.n))

def logicalIO (s : Shape) (c : Config) (rest : Algebra ℝ) (order : List Program) :
    LogicalProgramIO ℝ (SourceGraph.Node ℝ) (LogicalInputs s)
      (Fin s.batch → Fin s.m → Fin s.n → ℝ) where
  core := realIO s c rest order
  inputs := logicalInputsEquiv s
  outputs := logicalOutputsEquiv s

/-- Mathematical zero extension in the named batch/row/contraction axes. -/
def LogicalInputs.aAt {s : Shape} (xs : LogicalInputs s) (batch row k : Nat) : ℝ :=
  arrays 0 s ((logicalInputsEquiv s).symm xs) .a (s.a batch row k)
def LogicalInputs.bAt {s : Shape} (xs : LogicalInputs s) (batch k col : Nat) : ℝ :=
  arrays 0 s ((logicalInputsEquiv s).symm xs) .b (s.b batch k col)


end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Interface
