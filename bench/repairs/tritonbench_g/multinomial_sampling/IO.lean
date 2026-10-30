import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.KernelSource
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPBackend
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Spec
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.UndefProgramIO
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPReference
import VeriTile.Triton.Memory.LogicalDomainRelationIO
import Mathlib.Tactic.FinCases
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Real

/-! The selected FP32 source has four typed memory inputs and one signed
integer output. Arbitrary program IDs, original strides and input/output
aliases are retained. Even with zero tokens the initial label read is present,
so its logical window has at least one column. Undefined masked-load payloads
are separate logical inputs, supplied by the shared UndefProgramIO adapter. -/
namespace VeriTile.Bench.Repairs.MultinomialSampling.Interface
open VeriTile.Triton FP.Structural SourceValue
noncomputable section
variable {α : Type} [Inhabited α]

abbrev inputs (p : Parameters) : List IOChannel :=
  [⟨.real,[p.batches,p.tokens],p.scores⟩,
   ⟨.int,[p.batches],p.seeds⟩,
   ⟨.int,[p.batches],p.offsets⟩,
   ⟨.int,[p.batches,max p.tokens 1],p.indices⟩]

abbrev Inputs (α : Type) (p : Parameters) := ChannelValues α (inputs p)
abbrev Outputs (p : Parameters) := TileIndex [p.block] → Int

def inputOffset (p : Parameters) (i : Fin (inputs p).length) :
    TileIndex (inputs p)[i].shape → Nat :=
  match i with
  | ⟨0,_⟩ => fun j => j.1.val*p.strideScoreBatch+j.2.1.val*p.strideScoreToken
  | ⟨1,_⟩ | ⟨2,_⟩ => fun j => j.1.val
  | ⟨3,_⟩ => fun j => j.1.val*p.strideIndexBatch+j.2.1.val*p.strideIndexToken

def job (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat) : Invocation α where
  kernel := kernel p
  backend := backend rest P
  pids := fun _ => pid
  setPids := fun ax => decide (ax=0)

def memoryIO (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat) :
    ProgramIO α α where
  inputs := inputs p
  outputs := [⟨.int,[p.block],p.output⟩]
  scratch := []
  buffers := [p.scores,p.seeds,p.offsets,p.indices,p.output]
  read := fun i _ => inputOffset p i
  inputMask := fun _ _ _ => True
  write := fun | ⟨0,_⟩, _, j => pid*p.block+j.1.val
  outputMask := fun | ⟨0,_⟩, _, j => pid*p.block+j.1.val<p.batches
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [job rest P p pid]
  encode := id
  observe := fun _ _ => id

def score (p : Parameters) (vals : Inputs α p) (row token : Nat) : α :=
  if h : row<p.batches ∧ token<p.tokens then vals 0 (⟨row,h.1⟩,⟨token,h.2⟩,())
  else default

def seed (p : Parameters) (vals : Inputs α p) (row : Nat) : Int :=
  if h : row<p.batches then vals 1 (⟨row,h⟩,()) else 0

def offset (p : Parameters) (vals : Inputs α p) (row : Nat) : Int :=
  if h : row<p.batches then vals 2 (⟨row,h⟩,()) else 0

def label (p : Parameters) (vals : Inputs α p) (row token : Nat) : Int :=
  if h : row<p.batches ∧ token<max p.tokens 1 then vals 3 (⟨row,h.1⟩,⟨token,h.2⟩,())
  else 0

def weights (p : Parameters) (vals : Inputs ℝ p) (row j : Nat) : ℝ :=
  if j<p.tokens then score p vals row j else 0

/-- Independent categorical law on every active row of this source program.
Seed payload typing is a separate logical domain, including masked lanes. -/
def ProbabilityDomain (p : Parameters) (pid : Nat) (vals : Inputs ℝ p) : Prop :=
  ∀ i : TileIndex [p.block],pid*p.block+i.1.val<p.batches →
    Spec.Domain p.tokens (weights p vals (pid*p.block+i.1.val))

def categorical (p : Parameters) (pid : Nat) (vals : Inputs ℝ p)
    (outputs : Outputs p) : Prop :=
  ∀ i : TileIndex [p.block],pid*p.block+i.1.val<p.batches →
    Spec.Result p.tokens (weights p vals (pid*p.block+i.1.val))
      (label p vals (pid*p.block+i.1.val))
      (seed p vals (pid*p.block+i.1.val)) (offset p vals (pid*p.block+i.1.val)) (outputs i)

#specNonCircular categorical avoiding [kernel,source,FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.MultinomialSampling.Interface

namespace VeriTile.Bench.Repairs.MultinomialSampling.IOUndef
open VeriTile.Triton FP.Structural SourceValue
noncomputable section

def row (p : Parameters) (pid : Nat) (i : Row p) : Nat := pid * p.block + i.1.val

/-- The three masked loads without an explicit `other` retain their own
integer fallback payloads. Only inactive row lanes are observed. -/
def channels (p : Parameters) : List IOChannel :=
  [⟨.int, [p.block], p.seeds⟩, ⟨.int, [p.block], p.offsets⟩,
    ⟨.int, [p.block], p.indices⟩]

/-- Add the actual typed fallback observations to any concrete source-bound
memory IO. This changes neither the source schedule nor any memory channel.
No seed, offset, label, PID, or block divisibility is fixed here. -/
def addUndefined {α β : Type} (p : Parameters) (pid : Nat)
    (base : ProgramIO α β) : UndefProgramIO α β where
  toProgramIO := base
  undefInputs := channels p
  undefRead := fun u _ => Fin.cases (row p pid)
    (Fin.cases (row p pid)
      (Fin.cases (fun i => row p pid i * p.strideIndexBatch)
        (fun v => Fin.elim0 v))) u
  undefMask := fun u _ => Fin.cases (fun i => ¬ row p pid i < p.batches)
    (Fin.cases (fun i => ¬ row p pid i < p.batches)
      (Fin.cases (fun i => ¬ row p pid i < p.batches) (fun v => Fin.elim0 v))) u

variable {α β : Type} (p : Parameters) (pid : Nat) (base : ProgramIO α β)

def seed (vals : (addUndefined p pid base).Inputs)
    (activeSeed : (i : Row p) → row p pid i < p.batches → Int) (i : Row p) : Int :=
  if h : row p pid i < p.batches then activeSeed i h else vals.undefined (0 : Fin 3) i

/-- Exactly the original signed64 Philox seed domain, including padded rows.
There is no extra range condition on offsets or labels. -/
def SeedDomain (vals : (addUndefined p pid base).Inputs)
    (activeSeed : (i : Row p) → row p pid i < p.batches → Int) : Prop :=
  ∀ i, Random.IntegerType.i64.inRange (seed p pid base vals activeSeed i) = Bool.true

def offset (vals : (addUndefined p pid base).Inputs)
    (activeOffset : (i : Row p) → row p pid i < p.batches → Int) (i : Row p) : Int :=
  (BitVec.ofInt 32 (if h : row p pid i < p.batches then activeOffset i h
    else vals.undefined (1 : Fin 3) i)).toInt

def initialLabel (vals : (addUndefined p pid base).Inputs)
    (activeLabel : (i : Row p) → row p pid i < p.batches → Int) (i : Row p) : Int :=
  if h : row p pid i < p.batches then activeLabel i h else vals.undefined (2 : Fin 3) i

variable [Inhabited β]

/-- Every original state supplies these logical inputs. Values of unobserved
fallback addresses and all other dtypes are left completely unrestricted. -/
def extract (memory : base.Inputs) (s : State β) : (addUndefined p pid base).Inputs where
  memory := memory
  undefined := Fin.cases (fun i => s.undef .int p.seeds (row p pid i))
    (Fin.cases (fun i => s.undef .int p.offsets (row p pid i))
      (Fin.cases (fun i => s.undef .int p.indices (row p pid i * p.strideIndexBatch))
        (fun u => Fin.elim0 u)))

end
end VeriTile.Bench.Repairs.MultinomialSampling.IOUndef

namespace VeriTile.Bench.Repairs.MultinomialSampling.Interface
open VeriTile.Triton FP.Structural SourceValue
noncomputable section
variable {α : Type} [Inhabited α]

def io (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat) : UndefProgramIO α α :=
  IOUndef.addUndefined p pid (memoryIO rest P p pid)

/-- Logical inputs preserve arbitrary signed masked-load payloads. Memory and
fallback values are bound separately; none is manufactured by an initializer. -/
structure Data (α : Type) (p : Parameters) where
  memory : Inputs α p
  seedFallback : Row p → Int
  offsetFallback : Row p → Int
  labelFallback : Row p → Int

def data (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat)
    (vals : (io rest P p pid).Inputs) : Data α p :=
  ⟨vals.memory,vals.undefined (0 : Fin 3),vals.undefined (1 : Fin 3),vals.undefined (2 : Fin 3)⟩

namespace Logical
def row (p : Parameters) (pid : Nat) (i : Row p) : Nat := IOUndef.row p pid i
def seed (p : Parameters) (pid : Nat) (x : Data α p) (i : Row p) : Int :=
  if row p pid i<p.batches then Interface.seed p x.memory (row p pid i) else x.seedFallback i
def offset (p : Parameters) (pid : Nat) (x : Data α p) (i : Row p) : Int :=
  (BitVec.ofInt 32 (if row p pid i<p.batches then Interface.offset p x.memory (row p pid i)
    else x.offsetFallback i)).toInt
def initialLabel (p : Parameters) (pid : Nat) (x : Data α p) (i : Row p) : Int :=
  if row p pid i<p.batches then label p x.memory (row p pid i) 0 else x.labelFallback i

def SeedDomain (p : Parameters) (pid : Nat) (x : Data α p) : Prop :=
  ∀ i,Random.IntegerType.i64.inRange (seed p pid x i)=Bool.true

def sample (P : Primitives α) (p : Parameters) (pid : Nat) (x : Data α p) (i : Row p) : α :=
  P.bits32 ⟨(Random.rand (seed p pid x i) (offset p pid x i) .bits32 10).bits⟩

def scores (P : Primitives α) (p : Parameters) (pid : Nat) (x : Data α p)
    (b : Nat) (i : Lane p) : α :=
  P.cast (some .fp32) .real .real
    (if row p pid (i.1,())<p.batches ∧ b+i.2.1.val<p.tokens then
      score p x.memory (row p pid (i.1,())) (b+i.2.1.val)
    else P.zero (some .fp32) .real)

def scanned (P : Primitives α) (p : Parameters) (pid : Nat) (x : Data α p) (b : Nat) : Lane p → α :=
  P.scan none .sum ⟨1,by simp⟩ .forward (scores P p pid x b)
def cumulative (P : Primitives α) (p : Parameters) (pid : Nat) (x : Data α p)
    (b : Nat) (st : Accumulator α p) (i : Lane p) : α :=
  P.add (st.acc (i.1,())) (scanned P p pid x b i)
def valid (P : Primitives α) (p : Parameters) (pid : Nat) (x : Data α p)
    (b : Nat) (st : Accumulator α p) (i : Lane p) : Bool :=
  P.lt (P.sub (cumulative P p pid x b st i) (scores P p pid x b i)) (sample P p pid x (i.1,())) &&
    P.le (sample P p pid x (i.1,())) (cumulative P p pid x b st i)
def found (P : Primitives α) (p : Parameters) (pid : Nat) (x : Data α p)
    (b : Nat) (st : Accumulator α p) (i : Row p) : Bool :=
  decide (0<TileShape.discreteBoolSum ⟨1,by simp⟩ Bool.false (valid P p pid x b st) i)
def selected (P : Primitives α) (p : Parameters) (pid : Nat) (x : Data α p)
    (b : Nat) (st : Accumulator α p) (i : Row p) : Nat :=
  b+TileShape.discreteIntArgMax ⟨1,by simp⟩ (fun j=>if valid P p pid x b st j then (1:Int) else 0) i
def advance (P : Primitives α) (p : Parameters) (pid : Nat) (x : Data α p)
    (b : Nat) (st : Accumulator α p) : Accumulator α p where
  acc := fun i=>P.add (st.acc i) (P.maximum (some .fp32) ⟨1,by simp⟩ Bool.false (scanned P p pid x b) i)
  output := fun i=>if found P p pid x b st i then
    (if row p pid i<p.batches then
      label p x.memory (row p pid i) (min (selected P p pid x b st i) (p.tokens-1)) else -1)
    else st.output i
def initial (P : Primitives α) (p : Parameters) (pid : Nat) (x : Data α p) : Accumulator α p :=
  ⟨fun _=>P.zero (some .fp32) .real,initialLabel p pid x⟩
def rounds (P : Primitives α) (p : Parameters) (pid : Nat) (x : Data α p) : Nat → Accumulator α p
  | 0 => initial P p pid x
  | k+1 => advance P p pid x (k*p.blockTokens) (rounds P p pid x k)
end Logical

def SeedDomain (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat)
    (vals : (io rest P p pid).Inputs) : Prop :=
  Logical.SeedDomain p pid (data rest P p pid vals)

/-- The independent source-precision recurrence retains scan/max plans and
all native primitive operations. It is not a native-to-Real equality. -/
def nativeReference (P : Primitives α) (p : Parameters) (pid : Nat)
    (x : Data α p) : Outputs p :=
  fun i => (Logical.rounds P p pid x (count p)).output i

/-- Package the independent lane result into the one public signed output
channel. This adapter only reads logical input fields. -/
def nativeOutputs (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat)
    (vals : (io rest P p pid).Inputs) : (io rest P p pid).Outputs :=
  fun | ⟨0,_⟩, i=>nativeReference P p pid (data rest P p pid vals) i

#specNonCircular nativeReference avoiding
  [FP.Structural.exec,FP.Structural.run,SourceValue.rounds,SourceValue.written,kernel,source]
end
end VeriTile.Bench.Repairs.MultinomialSampling.Interface

namespace VeriTile.Bench.Repairs.MultinomialSampling.LogicalView
open VeriTile.Triton FP.Structural SourceValue Interface
noncomputable section
structure Inputs (p : Parameters) where
  probability : Fin p.batches → Fin p.tokens → ℝ
  seed : Fin p.batches → Int
  offset : Fin p.batches → Int
  label : Fin p.batches → Fin (max p.tokens 1) → Int
  seedFallback : Fin p.block → Int
  offsetFallback : Fin p.block → Int
  labelFallback : Fin p.block → Int

def inputEquiv (rest : Algebra ℝ) (p : Parameters) (pid : Nat) :
    (io rest Real.primitives p pid).Inputs ≃ Inputs p where
  toFun := fun xs => ⟨(fun b k => xs.memory ⟨0,by change 0<4; decide⟩ (b,k,())),(fun b => xs.memory ⟨1,by change 1<4; decide⟩ (b,())),
    (fun b => xs.memory ⟨2,by change 2<4; decide⟩ (b,())),(fun b k => xs.memory ⟨3,by change 3<4; decide⟩ (b,k,())),
    (fun i => xs.undefined ⟨0,by change 0<3; decide⟩ (i,())),(fun i => xs.undefined ⟨1,by change 1<3; decide⟩ (i,())),(fun i => xs.undefined ⟨2,by change 2<3; decide⟩ (i,()))⟩
  invFun := fun xs => ⟨
    Fin.cases (fun j => xs.probability j.1 j.2.1)
      (Fin.cases (fun j => xs.seed j.1) (Fin.cases (fun j => xs.offset j.1)
        (Fin.cases (fun j => xs.label j.1 j.2.1) (fun k => k.elim0)))),
    Fin.cases (fun j => xs.seedFallback j.1) (Fin.cases (fun j => xs.offsetFallback j.1)
      (Fin.cases (fun j => xs.labelFallback j.1) (fun k => k.elim0)))⟩
  left_inv := by
    intro xs
    cases xs with
    | mk memory undefined =>
      apply congrArg₂ (fun a b => ({memory:=a,undefined:=b} : (io rest Real.primitives p pid).Inputs))
      · funext o j; fin_cases o <;> rfl
      · funext o j; fin_cases o <;> rfl
  right_inv := by intro xs; cases xs; rfl

def outputEquiv (rest : Algebra ℝ) (p : Parameters) (pid : Nat) :
    (io rest Real.primitives p pid).Outputs ≃ (Fin p.block → Int) where
  toFun := fun ys i => ys ⟨0,by change 0<1; decide⟩ (i,())
  invFun := fun ys => Fin.cases (fun j => ys j.1) (fun k => k.elim0)
  left_inv := by intro ys; funext o j; fin_cases o; rfl
  right_inv := fun _ => rfl

/-- Logical lookup and source-domain transport, with arbitrary typed fallback
inputs kept separately. These maps do not supply or initialize an execution. -/
def weight (rest : Algebra ℝ) (p : Parameters) (pid : Nat) (xs : Inputs p) : Nat → Nat → ℝ :=
  weights p ((inputEquiv rest p pid).symm xs).memory

def seedValue (rest : Algebra ℝ) (p : Parameters) (pid : Nat) (xs : Inputs p) : Nat → Int :=
  Interface.seed p ((inputEquiv rest p pid).symm xs).memory

def offsetValue (rest : Algebra ℝ) (p : Parameters) (pid : Nat) (xs : Inputs p) : Nat → Int :=
  Interface.offset p ((inputEquiv rest p pid).symm xs).memory

def labelValue (rest : Algebra ℝ) (p : Parameters) (pid : Nat) (xs : Inputs p) : Nat → Nat → Int :=
  label p ((inputEquiv rest p pid).symm xs).memory

def accepted (rest : Algebra ℝ) (p : Parameters) (pid : Nat) (xs : Inputs p) : Prop :=
  Interface.SeedDomain rest Real.primitives p pid ((inputEquiv rest p pid).symm xs) ∧
  ProbabilityDomain p pid ((inputEquiv rest p pid).symm xs).memory

def samplingIO (rest : Algebra ℝ) (p : Parameters) (pid : Nat) :
    LogicalDomainRelationIO (io rest Real.primitives p pid).Inputs
      (io rest Real.primitives p pid).Outputs (Inputs p) (Fin p.block → Int)
      (io rest Real.primitives p pid).SatisfiesOn where
  inputs := inputEquiv rest p pid
  outputs := outputEquiv rest p pid
end
end VeriTile.Bench.Repairs.MultinomialSampling.LogicalView
