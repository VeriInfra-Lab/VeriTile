import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.token_softmax_bloom.Proofs.KernelCore
import bench.tritonbench_g.token_softmax_llama.Proofs.Real
import bench.tritonbench_g.token_softmax_llama.Proofs.Native
import VeriTile.Triton.Float.ProgramIO

/-! Public logical IO for the complete selected source schedule. Metadata ports
are the existing natural-index projection of the checked signed metadata ABI;
the unchanged Tensor bridge retains INT32 loading and widening. Payload and
output windows retain each head/token stride and every selected launch. -/
namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.SoftmaxIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
noncomputable section

def channels (α : Type) (p : Parameters) (work : List Grid.PID) :=
  ChannelValues α [⟨.nat, [work.length], Region.cast p.start⟩,
    ⟨.nat, [work.length], Region.cast p.length⟩, ⟨.fp32, [work.length,p.width], p.input⟩]

def start {α : Type} (p : Parameters) (work : List Grid.PID)
    (vals : channels α p work) (r : Fin work.length) : Nat := vals (0 : Fin 3) (r,())

def length {α : Type} (p : Parameters) (work : List Grid.PID)
    (vals : channels α p work) (r : Fin work.length) : Nat := vals (1 : Fin 3) (r,())

def inputOffset {α : Type} (p : Parameters) (work : List Grid.PID)
    (vals : channels α p work) (r : Fin work.length) (c : Nat) : Nat :=
  (work.get r).2*p.inputHeadStride+(start p work vals r+c)*p.inputTokenStride

def outputOffset {α : Type} (p : Parameters) (work : List Grid.PID)
    (vals : channels α p work) (r : Fin work.length) (c : Nat) : Nat :=
  (work.get r).2*p.outputHeadStride+(start p work vals r+c)*p.outputTokenStride

/-- Repeated identical launches remain admitted: uniqueness identifies
PID/lane events, not their positions in the schedule. -/
def Unique {α : Type} (p : Parameters) (work : List Grid.PID)
    (vals : channels α p work) : Prop :=
  ∀ r s (i j : Fin p.width), i.val < length p work vals r → j.val < length p work vals s →
    outputOffset p work vals r i.val = outputOffset p work vals s j.val →
      work.get r = work.get s ∧ i = j

def Cover {α : Type} (p : Parameters) (work : List Grid.PID)
    (vals : channels α p work) : Prop := ∀ r, length p work vals r ≤ p.width

def ports (α : Type) (p : Parameters) (work : List Grid.PID) : MemoryIO α where
  inputs := [⟨.nat, [work.length], Region.cast p.start⟩,
    ⟨.nat, [work.length], Region.cast p.length⟩, ⟨.fp32, [work.length,p.width], p.input⟩]
  outputs := [⟨.fp32, [work.length,p.width], p.output⟩]
  buffers := [Region.cast p.start, Region.cast p.length, p.input, p.output]
  read := by
    intro port vals
    refine Fin.cases (fun i => (work.get i.1).1*p.startStride) (fun next => ?_) port
    refine Fin.cases (fun i => (work.get i.1).1*p.lengthStride) (fun last => ?_) next
    have eq : last = (0 : Fin 1) := Fin.eq_zero last
    subst last
    exact fun i => inputOffset p work vals i.1 i.2.1.val
  inputMask := by
    intro port vals
    refine Fin.cases (fun _ => True) (fun next => ?_) port
    refine Fin.cases (fun _ => True) (fun last => ?_) next
    have eq : last = (0 : Fin 1) := Fin.eq_zero last
    subst last
    exact fun i => i.2.1.val < length p work vals i.1
  write := by
    intro port vals
    have eq : port = (0 : Fin 1) := Fin.eq_zero port
    subst port
    exact fun i => outputOffset p work vals i.1 i.2.1.val
  outputMask := by
    intro port vals
    have eq : port = (0 : Fin 1) := Fin.eq_zero port
    subst port
    exact fun i => i.2.1.val < length p work vals i.1
  scratchWrite := by intro port; exact Fin.elim0 port
  scratchMask := by intro port; exact Fin.elim0 port

def invocation {β : Type} (M : Algebra β) (p : Parameters) (pid : Grid.PID) : Invocation β where
  kernel := kernel .fp32 p
  backend := M
  clearRegisters := Bool.true
  pids := fun ax => if ax = 0 then pid.1 else if ax = 1 then pid.2 else 0
  numPids := some (fun _ => 0)

def io {α β : Type} (M : Algebra β) (encode : α → β) (p : Parameters)
    (observe : Fin p.width → β → α) (work : List Grid.PID) : ProgramIO α β :=
  { ports α p work with
    schedule := fun _ => work.map (invocation M p)
    encode := encode
    observe := by
      intro port
      have eq : port = (0 : Fin 1) := Fin.eq_zero port
      subst port
      exact fun i => observe i.2.1 }

/-- Independent finite softmax of the unpadded logical segment. -/
def realRow {B : Nat} (n : Nat) (xs : Fin B → ℝ) (i : Fin B) : ℝ :=
  Real.exp (xs i) / ∑ j : Fin B, if j.val < n then Real.exp (xs j) else 0

/-- Existing ordered native instruction graph; all FP32 words are admitted. -/
def nativeRow (I : Native.Instructions) (V B n : Nat) (xs : Fin B → Native.Word)
    (i : Fin B) : Native.Word :=
  SoftmaxTriton1.NativeGraph.row I V n B (fun j => if h : j < B then xs ⟨j,h⟩ else default) i.val

def realIO (p : Parameters) (work : List Grid.PID) : ProgramIO ℝ Real.Scalar :=
  io Real.algebra (fun x => some x) p (fun _ => WithBot.unbotD 0) work

def nativeIO (I : Native.Instructions) (V : Nat) (p : Parameters) (work : List Grid.PID) :
    ProgramIO Native.Word Native.Node :=
  io (SoftmaxTriton1.NativeGraph.algebra I V) SoftmaxTriton1.NativeGraph.atom p
    (fun i value => value (SoftmaxTriton1.NativeGraph.owner V p.width i.val)) work

def target {α β : Type} (M : Algebra β) (encode : α → β) (p : Parameters)
    (observe : Fin p.width → β → α) (work : List Grid.PID)
    (f : Nat → (Fin p.width → α) → Fin p.width → α) :
    (io M encode p observe work).Inputs → (io M encode p observe work).Outputs := by
  intro vals port
  have eq : port = (0 : Fin 1) := Fin.eq_zero port
  subst port
  exact fun i => f (length p work vals i.1) (fun c => vals (2 : Fin 3) (i.1,c,())) i.2.1

def realDomain (p : Parameters) (work : List Grid.PID) (vals : (realIO p work).Inputs) : Prop :=
  Unique p work vals ∧ Cover p work vals

end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.SoftmaxIO

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.Logical
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
noncomputable section

structure Inputs (p : Parameters) (work : List Grid.PID) where
  start : Fin work.length → Nat
  length : Fin work.length → Nat
  values : Fin work.length → Fin p.width → ℝ

def inputs (p : Parameters) (work : List Grid.PID) :
    (SoftmaxIO.realIO p work).Inputs ≃ Inputs p work where
  toFun := fun values => ⟨SoftmaxIO.start p work values, SoftmaxIO.length p work values,
    fun job col => values ⟨2, by simp [SoftmaxIO.realIO,SoftmaxIO.io,SoftmaxIO.ports]⟩ (job,col,())⟩
  invFun := fun xs port => match port with
    | ⟨0,_⟩ => fun j => xs.start j.1
    | ⟨1,_⟩ => fun j => xs.length j.1
    | ⟨2,_⟩ => fun j => xs.values j.1 j.2.1
    | ⟨n+3,h⟩ => False.elim (by simp [SoftmaxIO.realIO,SoftmaxIO.io,SoftmaxIO.ports] at h)
  left_inv := by
    intro values
    funext port
    fin_cases port <;> funext j
    all_goals first | (rcases j with ⟨a,b,⟨⟩⟩; rfl) | (rcases j with ⟨a,⟨⟩⟩; rfl)
  right_inv := by intro xs; cases xs; rfl

def outputs (p : Parameters) (work : List Grid.PID) :
    (SoftmaxIO.realIO p work).Outputs ≃ (Fin work.length → Fin p.width → ℝ) where
  toFun := fun values job col => values ⟨0, by simp [SoftmaxIO.realIO,SoftmaxIO.io,SoftmaxIO.ports]⟩ (job,col,())
  invFun := fun f port => match port with
    | ⟨0,_⟩ => fun j => f j.1 j.2.1
    | ⟨n+1,h⟩ => False.elim (by simp [SoftmaxIO.realIO,SoftmaxIO.io,SoftmaxIO.ports] at h)
  left_inv := by
    intro values
    funext port
    fin_cases port
    funext j
    rcases j with ⟨a,b,⟨⟩⟩
    rfl
  right_inv := by intro f; rfl

/-- Total lookup outside the finite port. Coverage makes this extension unused in each logical sum. -/
def Inputs.at {p : Parameters} {work : List Grid.PID} (xs : Inputs p work)
    (job : Fin work.length) (col : Nat) : ℝ :=
  if h : col < p.width then xs.values job ⟨col,h⟩ else 0

/-- Exactly the old uniqueness and coverage domain transported through the input equivalence. -/
def domain (p : Parameters) (work : List Grid.PID) (xs : Inputs p work) : Prop :=
  SoftmaxIO.realDomain p work ((inputs p work).symm xs)

def io (p : Parameters) (work : List Grid.PID) :
    LogicalProgramIO ℝ Real.Scalar (Inputs p work) (Fin work.length → Fin p.width → ℝ) where
  core := SoftmaxIO.realIO p work
  inputs := inputs p work
  outputs := outputs p work

end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.Logical
