import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Base
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence
noncomputable section

namespace Source
structure State (α : Type) where
  memory : Memory α
  carry : Tile α

/-- Initialization reads seed before the first output store. No kernel is
launched for N=0. Same-format FP32 casts are word identities. -/
def forwardInit (ops : Ops α) (hasSeed : Bool) (p : Nat) (m : Memory α) : State α :=
  let carry := fun j => if hasSeed then m .seed p 0 j else ops.zero
  ⟨Memory.tile m .output p 0 carry,carry⟩

def forwardStep (ops : Ops α) (c : Shape) (p t : Nat) (s : State α) : State α :=
  let carry := fun j => ops.fma (s.carry j) (s.memory .gate (p/c.tiles) t 0) (s.memory .x p t j)
  ⟨Memory.tile s.memory .output p (t+1) carry,carry⟩

def forwardLoop (ops : Ops α) (c : Shape) (hasSeed : Bool) (p : Nat) (m : Memory α) : Nat → State α
  | 0 => forwardInit ops hasSeed p m
  | t+1 => forwardStep ops c p t (forwardLoop ops c hasSeed p m t)

def forwardKernel (ops : Ops α) (c : Shape) (hasSeed : Bool) (p : Nat) (m : Memory α) : Memory α :=
  if c.N=0 then m else (forwardLoop ops c hasSeed p m (c.N-1)).memory

def forwardGrid (ops : Ops α) (c : Shape) (hasSeed : Bool) (m : Memory α) : Nat → Memory α
  | 0 => m
  | p+1 => forwardKernel ops c hasSeed p (forwardGrid ops c hasSeed m p)

/-- Full reverse loop body: all reads precede the DG and DI stores. S is the
same current output allocation produced by the forward grid. -/
def backwardStep (ops : Ops α) (c : Shape) (trees : Trees c) (p count : Nat) (s : State α) : State α :=
  let t := c.N-1-count
  let carry := fun j => ops.fma (s.carry j) (s.memory .gate (p/c.tiles) t 0) (s.memory .incoming p t j)
  let dg := Reduction.product ops (trees.tile p (t-1))
    (fun j => (carry j,s.memory .output p (t-1) j))
  let m := Memory.scalar s.memory .tileGate p (t-1) dg
  ⟨Memory.tile m .dx p (t-1) carry,carry⟩

def backwardLoop (ops : Ops α) (c : Shape) (trees : Trees c) (p : Nat) (m : Memory α) : Nat → State α
  | 0 => ⟨m,fun _ => ops.zero⟩
  | n+1 => backwardStep ops c trees p n (backwardLoop ops c trees p m n)

/-- The separate DL postlude also executes at N=1; its zero-times-gate FMA
is retained in the native graph, even though it erases in Real arithmetic. -/
def backwardPost (ops : Ops α) (c : Shape) (p : Nat) (s : State α) : Memory α :=
  let carry := fun j => ops.fma (s.carry j) (s.memory .gate (p/c.tiles) 0 0) (s.memory .incoming p 0 j)
  Memory.tile s.memory .dseed p 0 carry

def backwardKernel (ops : Ops α) (c : Shape) (trees : Trees c) (p : Nat) (m : Memory α) : Memory α :=
  if c.N=0 then m else backwardPost ops c p (backwardLoop ops c trees p m (c.N-1))

def backwardGrid (ops : Ops α) (c : Shape) (trees : Trees c) (m : Memory α) : Nat → Memory α
  | 0 => m
  | p+1 => backwardKernel ops c trees p (backwardGrid ops c trees m p)

/-- Python's two complete last-slice zero assignments, after every program. -/
def zeroTail (ops : Ops α) (c : Shape) (m : Memory α) : Memory α :=
  fun r p t j => if p<c.programs ∧ t=c.N-1 ∧ c.N>0 ∧
    ((r=.dx ∧ j<1024) ∨ (r=.tileGate ∧ j=0)) then ops.zero else m r p t j

/-- Torch view(B,H,N,-1).sum(-1): each output reads the current, host-completed
DG scratch and evaluates a complete scalar-addition tree, including its zeros. -/
def hostSum (ops : Ops α) (c : Shape) (trees : Trees c) (m : Memory α) : Memory α :=
  fun r h t j => if r=.dgate ∧ h<c.heads ∧ t<c.N ∧ j=0 then
    Reduction.eval ops (trees.host h t) (fun q => m .tileGate (h*c.tiles+q) t 0)
    else m r h t j

def emptyBackward (ops : Ops α) (c : Shape) (hasSeed : Bool) (m : Memory α) : Memory α :=
  fun r p t j => if hasSeed ∧ r=.dseed ∧ p<c.programs ∧ t=0 ∧ j<1024 then ops.zero else m r p t j

def run (ops : Ops α) (c : Shape) (trees : Trees c) (hasSeed : Bool) (m : Memory α) : Memory α :=
  if c.N=0 then emptyBackward ops c hasSeed m else
    let f := forwardGrid ops c hasSeed m c.programs
    let b := backwardGrid ops c trees f c.programs
    hostSum ops c trees (zeroTail ops c b)
end Source

#specNonCircular Reference.forward avoiding [Source.run,Source.forwardLoop]
#specNonCircular Reference.reverse avoiding [Source.run,Source.backwardLoop]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence
