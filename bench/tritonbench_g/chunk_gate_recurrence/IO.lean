import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelProgram
import VeriTile.Triton.Float.ObservedProgramIO
import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelRealReference

/-! One complete source program, including its time loop. Channels retain the
actual Structural dtype; the shared source fixes FP32 load/computation/store
precision. The three program IDs and four optional seed strides are explicit.
The backward gate channel includes the last gate, even for a zero carry. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.IO
open VeriTile.Triton FP.Structural KernelProgram
noncomputable section

def launch (M : Algebra α) (kind : Kind) (c : Shape) (s : SeedStrides)
    (seed : Bool) (p : Program c) : Invocation α where
  kernel := source kind c s seed
  backend := FP.ProfiledRewrite.engine M
  pids := fun ax => if ax=0 then p.head else if ax=1 then p.key else p.value
  setPids := fun ax => decide (ax<3)

def forward (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) : ProgramIO α α where
  inputs := [⟨.real,[c.N-1,64,16],"S"⟩,⟨.real,[c.N-1],"D"⟩,
    ⟨.real,[64,16],"Seed"⟩]
  outputs := [⟨.real,[c.N,64,16],"O"⟩]
  scratch := []
  buffers := ["S","D","Seed","O"]
  read := fun i => Fin.cases (fun _ j => address c p j.1.val j.2.1.val j.2.2.1.val)
    (Fin.cases (fun _ j => gateAddress c p j.1.val)
      (Fin.cases (fun _ j => seedAddress c p s j.1.val j.2.1.val) (fun i => Fin.elim0 i))) i
  inputMask := fun i => Fin.cases (fun _ _ => True)
    (Fin.cases (fun _ _ => True) (Fin.cases (fun _ _ => seed=Bool.true) (fun i => Fin.elim0 i))) i
  write := fun o => Fin.cases (fun _ j => address c p j.1.val j.2.1.val j.2.2.1.val)
    (fun i => Fin.elim0 i) o
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => [launch M .forward c s seed p]
  encode := id
  observe := fun _ _ => id

def backward (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) : ProgramIO α α where
  inputs := [⟨.fp32,[c.N-1,64,16],"S"⟩,⟨.real,[c.N],"D"⟩,
    ⟨.real,[c.N,64,16],"DS"⟩]
  outputs := [⟨.real,[c.N-1,64,16],"DI"⟩,⟨.real,[c.N-1],"DG"⟩,
    ⟨.fp32,[64,16],"DL"⟩]
  scratch := []
  buffers := ["S","D","DS","DI","DG","DL"]
  read := fun i => Fin.cases (fun _ j => address c p j.1.val j.2.1.val j.2.2.1.val)
    (Fin.cases (fun _ j => gateAddress c p j.1.val)
      (Fin.cases (fun _ j => address c p j.1.val j.2.1.val j.2.2.1.val) (fun i => Fin.elim0 i))) i
  inputMask := fun _ _ _ => True
  write := fun o => Fin.cases (fun _ j => address c p j.1.val j.2.1.val j.2.2.1.val)
    (Fin.cases (fun _ j => gradientAddress c p j.1.val)
      (Fin.cases (fun _ j => dseedAddress c p j.1.val j.2.1.val) (fun i => Fin.elim0 i))) o
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => [launch M .backward c s seed p]
  encode := id
  observe := fun _ _ => id

def program (M : Algebra α) (kind : Kind) (c : Shape) (s : SeedStrides)
    (seed : Bool) (p : Program c) : ProgramIO α α :=
  match kind with
  | .forward => forward M c s seed p
  | .backward => backward M c s seed p

/-- Total extension used only by input-only formulas. Proofs show that all
queried time indices lie within the corresponding finite channel. -/
def time [Inhabited α] (xs : Fin n → α) (t : Nat) : α :=
  if h : t<n then xs ⟨t,h⟩ else default

end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.IO

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Reference
open VeriTile.Triton FP.Structural KernelProgram
open scoped BigOperators
noncomputable section
variable {α : Type} [Inhabited α]

def forwardInputs {M : Algebra α} {c : Shape} {s : SeedStrides} {seed : Bool} {p : Program c}
    (a : (IO.forward M c s seed p).Inputs) : KernelReference.Inputs α where
  source := fun t i => IO.time (fun t => a (0 : Fin 3) (t,i)) t
  gate := fun t => IO.time (fun t => a (1 : Fin 3) (t,PUnit.unit)) t
  seed := fun i => a (2 : Fin 3) i
  incoming := fun _ _ => default
  saved := fun _ _ => default

def backwardInputs {M : Algebra α} {c : Shape} {s : SeedStrides} {seed : Bool} {p : Program c}
    (a : (IO.backward M c s seed p).Inputs) : KernelReference.Inputs α where
  source := fun _ _ => default
  gate := fun t => IO.time (fun t => a (1 : Fin 3) (t,PUnit.unit)) t
  seed := fun _ => default
  incoming := fun t i => IO.time (fun t => a (2 : Fin 3) (t,i)) t
  saved := fun t i => IO.time (fun t => a (0 : Fin 3) (t,i)) t

end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Reference

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.LogicalTensors
open VeriTile.Triton FP.Structural KernelProgram
noncomputable section

/- Coordinate equivalences retain all finite cells, the final backward gate,
and the optional forward seed even when its source input mask is inactive. -/
structure ForwardInputs (c : Shape) where
  source : Fin (c.N-1) → Fin (64) → Fin (16) → ℝ
  gate : Fin (c.N-1) → ℝ
  initial : Fin (64) → Fin (16) → ℝ

structure BackwardInputs (c : Shape) where
  saved : Fin (c.N-1) → Fin (64) → Fin (16) → ℝ
  gate : Fin (c.N) → ℝ
  incoming : Fin (c.N) → Fin (64) → Fin (16) → ℝ

structure ForwardOutputs (c : Shape) where
  state : Fin (c.N) → Fin (64) → Fin (16) → ℝ

structure BackwardOutputs (c : Shape) where
  state : Fin (c.N-1) → Fin (64) → Fin (16) → ℝ
  gate : Fin (c.N-1) → ℝ
  initial : Fin (64) → Fin (16) → ℝ

def forwardInputs (c : Shape) : ChannelValues ℝ [⟨.real,[c.N-1,64,16],"S"⟩,⟨.real,[c.N-1],"D"⟩,⟨.real,[64,16],"Seed"⟩] ≃ ForwardInputs c where
  toFun := fun values => ⟨Values.curryEquiv ℝ .real [c.N-1,64,16] (values (0 : Fin 3)),
    Values.curryEquiv ℝ .real [c.N-1] (values (1 : Fin 3)),
    Values.curryEquiv ℝ .real [64,16] (values (2 : Fin 3))⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (Values.curryEquiv ℝ .real [c.N-1,64,16]).symm values.source
    | ⟨1,_⟩ => (Values.curryEquiv ℝ .real [c.N-1]).symm values.gate
    | ⟨2,_⟩ => (Values.curryEquiv ℝ .real [64,16]).symm values.initial
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;> exact congrFun (Equiv.symm_apply_apply _ _) j
  right_inv := by
    intro values
    cases values
    dsimp
    congr 1 <;> apply Equiv.apply_symm_apply

def backwardInputs (c : Shape) : ChannelValues ℝ [⟨.fp32,[c.N-1,64,16],"S"⟩,⟨.real,[c.N],"D"⟩,⟨.real,[c.N,64,16],"DS"⟩] ≃ BackwardInputs c where
  toFun := fun values => ⟨Values.curryEquiv ℝ .fp32 [c.N-1,64,16] (values (0 : Fin 3)),
    Values.curryEquiv ℝ .real [c.N] (values (1 : Fin 3)),
    Values.curryEquiv ℝ .real [c.N,64,16] (values (2 : Fin 3))⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (Values.curryEquiv ℝ .fp32 [c.N-1,64,16]).symm values.saved
    | ⟨1,_⟩ => (Values.curryEquiv ℝ .real [c.N]).symm values.gate
    | ⟨2,_⟩ => (Values.curryEquiv ℝ .real [c.N,64,16]).symm values.incoming
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;> exact congrFun (Equiv.symm_apply_apply _ _) j
  right_inv := by
    intro values
    cases values
    dsimp
    congr 1 <;> apply Equiv.apply_symm_apply

def forwardOutputs (c : Shape) : ChannelValues ℝ [⟨.real,[c.N,64,16],"O"⟩] ≃ ForwardOutputs c where
  toFun := fun values => ⟨Values.curryEquiv ℝ .real [c.N,64,16] (values (0 : Fin 1))⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (Values.curryEquiv ℝ .real [c.N,64,16]).symm values.state
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;> exact congrFun (Equiv.symm_apply_apply _ _) j
  right_inv := by
    intro values
    cases values
    dsimp
    congr 1 <;> apply Equiv.apply_symm_apply

def backwardOutputs (c : Shape) : ChannelValues ℝ [⟨.real,[c.N-1,64,16],"DI"⟩,⟨.real,[c.N-1],"DG"⟩,⟨.fp32,[64,16],"DL"⟩] ≃ BackwardOutputs c where
  toFun := fun values => ⟨Values.curryEquiv ℝ .real [c.N-1,64,16] (values (0 : Fin 3)),
    Values.curryEquiv ℝ .real [c.N-1] (values (1 : Fin 3)),
    Values.curryEquiv ℝ .fp32 [64,16] (values (2 : Fin 3))⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (Values.curryEquiv ℝ .real [c.N-1,64,16]).symm values.state
    | ⟨1,_⟩ => (Values.curryEquiv ℝ .real [c.N-1]).symm values.gate
    | ⟨2,_⟩ => (Values.curryEquiv ℝ .fp32 [64,16]).symm values.initial
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;> exact congrFun (Equiv.symm_apply_apply _ _) j
  right_inv := by
    intro values
    cases values
    dsimp
    congr 1 <;> apply Equiv.apply_symm_apply

def Inputs (kind : Kind) (c : Shape) : Type := match kind with
  | .forward => ForwardInputs c
  | .backward => BackwardInputs c

def Outputs (kind : Kind) (c : Shape) : Type := match kind with
  | .forward => ForwardOutputs c
  | .backward => BackwardOutputs c

def inputs (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) :
    (IO.program KernelRealReference.algebra kind c s seed p).Inputs ≃ Inputs kind c := by
  cases kind
  · exact forwardInputs c
  · exact backwardInputs c

def outputs (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) :
    (IO.program KernelRealReference.algebra kind c s seed p).Outputs ≃ Outputs kind c := by
  cases kind
  · exact forwardOutputs c
  · exact backwardOutputs c

def forwardArrays (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) (a : ForwardInputs c) :=
  Reference.forwardInputs (M := KernelRealReference.algebra) (s := s) (seed := seed) (p := p)
    ((forwardInputs c).symm a)

def backwardArrays (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) (a : BackwardInputs c) :=
  Reference.backwardInputs (M := KernelRealReference.algebra) (s := s) (seed := seed) (p := p)
    ((backwardInputs c).symm a)

def io (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) :
    LogicalProgramIO ℝ ℝ (Inputs kind c) (Outputs kind c) where
  core := IO.program KernelRealReference.algebra kind c s seed p
  inputs := inputs kind c s seed p
  outputs := outputs kind c s seed p

end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.LogicalTensors
