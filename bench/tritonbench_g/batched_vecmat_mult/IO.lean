import bench.tritonbench_g.batched_vecmat_mult.Kernels
import bench.tritonbench_g.batched_vecmat_mult.Proofs.KernelProfile
import bench.tritonbench_g.batched_vecmat_mult.Proofs.Mathematical_SourceGraphTensor
import VeriTile.Triton.Float.ObservedProgramIO
import VeriTile.Triton.Float.AddressedInputsIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Memory.LogicalIO
import Mathlib.Tactic.FinCases

/-! Complete floor-grid IO of the selected FP32 source. Structural `.real`
channels carry its FP32 operation graph; source load/arithmetic/store precision
is fixed by the typed source, not by these interpreter channel tags. -/
namespace VeriTile.Bench.Repairs.BatchedVecmatMult.VecmatIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open SourceGraph (Expr)
noncomputable section

def inputs (p : Parameters) : List IOChannel :=
  [⟨.real,[p.M,p.K],p.A⟩,⟨.real,[p.M,p.N,p.K],p.B⟩]
def outputs (p : Parameters) : List IOChannel := [⟨.real,[p.M,p.N],p.output⟩]

def memoryIO (α : Type) (p : Parameters) : MemoryIO α where
  inputs := inputs p
  outputs := outputs p
  buffers := [p.A,p.B,p.output]
  read := by
    refine Fin.cases ?_ (Fin.cases ?_ (fun i => Fin.elim0 i))
    · exact fun _ j => j.1.val*p.K+j.2.1.val
    · exact fun _ j => j.1.val*p.N*p.K+j.2.1.val*p.K+j.2.2.1.val
  inputMask := fun _ _ _ => True
  write := fun o _ => by
    change Fin 1 at o
    have ho : o = (0 : Fin 1) := Subsingleton.elim _ _
    subst o
    exact fun j => j.1.val*p.N+j.2.1.val
  outputMask := fun _ _ _ => True
  scratchWrite := by intro i; exact Fin.elim0 i
  scratchMask := by intro i; exact Fin.elim0 i

def launch (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) (pid : Nat) : Invocation Expr where
  kernel := source p
  backend := SourceGraph.engine rest (KernelProfile.toPlans profile)
  clearRegisters := Bool.true
  pids := fun ax => if ax=0 then pid % SourceGraph.gridM p
    else if ax=1 then pid / SourceGraph.gridM p else 0
  numPids := some (fun ax => if ax=0 then SourceGraph.gridM p
    else if ax=1 then SourceGraph.gridN p else 1)

def schedule (rest : Algebra Expr) (profile : KernelProfile.Selector) (p : Parameters) :
    List (Invocation Expr) := (List.range (SourceGraph.gridCount p)).map (launch rest profile p)

/-- Arbitrary source-graph inputs retain the exact prior mathematical domain. -/
def graphIO (rest : Algebra Expr) (profile : KernelProfile.Selector) (p : Parameters) :
    ProgramIO Expr Expr where
  toMemoryIO := memoryIO Expr p
  schedule := fun _ => schedule rest profile p
  encode := id
  observe := fun _ _ => id

/-- Native input labels retain their actual source region and element offset.
The observer contracts the captured graph, then rounds the retained scalar
FP32 operations under the original instruction model. -/
def nativeIO (cfg : FP.Config) (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) : AddressedProgramIO ValueProofs.Word Expr where
  toMemoryIO := memoryIO ValueProofs.Word p
  schedule := fun _ => schedule rest profile p
  encodeAt := Expr.input
  observe := fun _ _ e => SourceGraph.evaluate cfg (SourceGraph.contract e)

def aValue {α : Type} (p : Parameters) (encode : RegionName → Nat → α → Expr)
    (vals : ChannelValues α (inputs p)) (m k : Nat) : Expr :=
  if hm : m<p.M then if hk : k<p.K then
    encode p.A (m*p.K+k) (vals (0 : Fin 2) (⟨m,hm⟩,⟨k,hk⟩,())) else .zero else .zero
def bValue {α : Type} (p : Parameters) (encode : RegionName → Nat → α → Expr)
    (vals : ChannelValues α (inputs p)) (m n k : Nat) : Expr :=
  if hm : m<p.M then if hn : n<p.N then if hk : k<p.K then
    encode p.B (m*p.N*p.K+n*p.K+k)
      (vals (1 : Fin 2) (⟨m,hm⟩,⟨n,hn⟩,⟨k,hk⟩,())) else .zero else .zero else .zero

/-- Independent source-order block recurrence, including reduction tree choice,
initial +0 and every carry addition. It reads logical inputs only. -/
def block (plans : SourceGraph.PlanSelector) (p : Parameters)
    (a : Nat → Nat → Expr) (b : Nat → Nat → Nat → Expr) (m n c : Nat) : Expr :=
  let args := fun k : Fin p.BK => Expr.mul (a m (c*p.BK+k.val)) (b m n (c*p.BK+k.val))
  SourceGraph.treeValue args (plans p.BK args).tree
def accumulate (plans : SourceGraph.PlanSelector) (p : Parameters)
    (a : Nat → Nat → Expr) (b : Nat → Nat → Nat → Expr) (m n : Nat) : Nat → Expr
  | 0 => .zero
  | c+1 => .add (accumulate plans p a b m n c) (block plans p a b m n c)

def graphTarget (profile : KernelProfile.Selector) (p : Parameters)
    (vals : ChannelValues Expr (inputs p)) : ChannelValues Expr (outputs p) := by
  intro o
  change Fin 1 at o
  have ho : o = (0 : Fin 1) := Subsingleton.elim _ _
  subst o
  exact fun j => accumulate (KernelProfile.toPlans profile) p
    (aValue p (fun _ _ => id) vals) (bValue p (fun _ _ => id) vals)
    j.1.val j.2.1.val (p.K/p.BK)

/-- Independent full-K mathematical dot product. -/
def realTarget (decode : ValueProofs.Word → ℝ) (p : Parameters)
    (vals : ChannelValues Expr (inputs p)) : ChannelValues ℝ (outputs p) := by
  intro o
  change Fin 1 at o
  have ho : o = (0 : Fin 1) := Subsingleton.elim _ _
  subst o
  change TileIndex [p.M,p.N] → ℝ
  exact fun j => ∑ k : Fin p.K,
    SourceGraph.realValue decode (vals (0 : Fin 2) (j.1,k,())) *
    SourceGraph.realValue decode (vals (1 : Fin 2) (j.1,j.2.1,k,()))

def nativeTarget (cfg : FP.Config) (profile : KernelProfile.Selector) (p : Parameters)
    (vals : ChannelValues ValueProofs.Word (inputs p)) : ChannelValues ValueProofs.Word (outputs p) := by
  intro o
  change Fin 1 at o
  have ho : o = (0 : Fin 1) := Subsingleton.elim _ _
  subst o
  exact fun j => SourceGraph.evaluate cfg (SourceGraph.contract
    (accumulate (KernelProfile.toPlans profile) p (aValue p Expr.input vals)
      (bValue p Expr.input vals) j.1.val j.2.1.val (p.K/p.BK)))

def realObservation (decode : ValueProofs.Word → ℝ) (p : Parameters) :
    (o : Fin (outputs p).length) → TileIndex (outputs p)[o].shape → Expr → ℝ :=
  fun _ _ e => SourceGraph.realValue decode (SourceGraph.contract e)

def nativeObservation (cfg : FP.Config) (p : Parameters) :
    (o : Fin (outputs p).length) → TileIndex (outputs p)[o].shape → Expr → ValueProofs.Word :=
  fun _ _ e => SourceGraph.evaluate cfg (SourceGraph.contract e)

/-- Observe the independent logical recurrence under the unchanged native
primitive model, without reading source execution or output memory. -/
def nativeGraphTarget (cfg : FP.Config) (profile : KernelProfile.Selector) (p : Parameters)
    (vals : ChannelValues Expr (inputs p)) : ChannelValues ValueProofs.Word (outputs p) :=
  fun o j => mapValue (nativeObservation cfg p o j) (outputs p)[o].dtype (graphTarget profile p vals o j)

end
end VeriTile.Bench.Repairs.BatchedVecmatMult.VecmatIO

/-! Candidate program data shares the same canonical memory windows. -/
namespace VeriTile.Bench.Repairs.BatchedVecmatMult.PairIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open SourceGraph (Expr)
open VecmatIO
noncomputable section

/-- Evaluation preserves the complete FP32 graph's explicit contraction pass.
Wrapping the observed word retains the arbitrary-expression logical inputs. -/
def observe (cfg : FP.Config) (e : Expr) : Expr :=
  .word (SourceGraph.evaluate cfg (SourceGraph.contract e))

def io (cfg : FP.Config) (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) : ProgramIO Expr Expr :=
  { graphIO rest profile p with observe := fun _ _ => observe cfg }

def launch (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) (pid : Nat) : Invocation Expr :=
  { VecmatIO.launch rest profile p pid with kernel := blockFirst p }

def candidate (cfg : FP.Config) (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) : ProgramIO.Variant (io cfg rest profile p) where
  schedule := fun _ => (List.range (SourceGraph.gridCount p)).map (launch rest profile p)
  observe := fun _ _ => observe cfg

end
end VeriTile.Bench.Repairs.BatchedVecmatMult.PairIO

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.VecmatIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open SourceGraph (Expr)
noncomputable section

structure LogicalInputs (p : Parameters) where
  a : Fin p.M → Fin p.K → Expr
  b : Fin p.M → Fin p.N → Fin p.K → Expr

def logicalInputsEquiv (p : Parameters) :
    ChannelValues Expr (inputs p) ≃ LogicalInputs p where
  toFun := fun vals => ⟨fun m k => vals (0 : Fin 2) (m,k,()), fun m n k => vals (1 : Fin 2) (m,n,k,())⟩
  invFun := fun xs => Fin.cases (fun j => xs.a j.1 j.2.1)
    (Fin.cases (fun j => xs.b j.1 j.2.1 j.2.2.1) (fun i => Fin.elim0 i))
  left_inv := by
    intro vals
    funext i j
    fin_cases i
    · rcases j with ⟨m,k,⟨⟩⟩; rfl
    · rcases j with ⟨m,n,k,⟨⟩⟩; rfl
  right_inv := by intro xs; cases xs; rfl

def logicalOutputsEquiv (p : Parameters) :
    ChannelValues ℝ (outputs p) ≃ (Fin p.M → Fin p.N → ℝ) where
  toFun := fun vals m n => vals (0 : Fin 1) (m,n,())
  invFun := fun f => Fin.cases (fun j => f j.1 j.2.1) (fun i => Fin.elim0 i)
  left_inv := by
    intro vals
    funext i j
    fin_cases i
    rcases j with ⟨m,n,⟨⟩⟩
    rfl
  right_inv := fun _ => rfl

/-- Arbitrary source expressions, the exact contracted Real observer and the
complete floor-grid schedule are retained by this coordinate view. -/
def logicalIO (decode : ValueProofs.Word → ℝ) (rest : Algebra Expr)
    (profile : KernelProfile.Selector) (p : Parameters) :
    LogicalIO (graphIO rest profile p).Inputs (ChannelValues ℝ (outputs p))
      (LogicalInputs p) (Fin p.M → Fin p.N → ℝ)
      ((graphIO rest profile p).Observes (realObservation decode p)) where
  inputs := logicalInputsEquiv p
  outputs := logicalOutputsEquiv p

end
end VeriTile.Bench.Repairs.BatchedVecmatMult.VecmatIO
