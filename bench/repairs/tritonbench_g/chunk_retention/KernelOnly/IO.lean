import bench.repairs.tritonbench_g.chunk_retention.Proofs.KernelNativePacket
import VeriTile.Triton.Float.AddressedInputsIO
import VeriTile.Triton.Float.LogicalProgramIO

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
noncomputable section

/-- Full logical inputs of a standalone stage. Saved histories are ordinary
inputs to forward-output and backward-QKV; this interface does not silently
replace the selected four standalone kernels by a host pipeline. -/
@[reducible] def inputs (c : Shape) : List IOChannel :=
  [⟨.real,[c.B*c.H,c.T,c.K],"q"⟩,⟨.real,[c.B*c.H,c.T,c.K],"k"⟩,
   ⟨.real,[c.B*c.H,c.T,c.V],"v"⟩,⟨.real,[c.B*c.H,c.T,c.V],"incoming"⟩,
   ⟨.real,[c.B*c.H,c.K,c.V],"h0"⟩,⟨.real,[c.B*c.H,c.K,c.V],"terminal"⟩,
   ⟨.real,[c.B*c.H,c.chunks,c.K,c.V],"h"⟩,
   ⟨.real,[c.B*c.H,c.chunks,c.K,c.V],"dh"⟩]

/-- Every original active tile output is represented: forward history/final
state/output, reverse history, dQ/dK and all value-block partial dV stores. -/
@[reducible] def outputs (c : Shape) : List IOChannel :=
  [⟨.real,[c.chunks,c.BK,c.BV],"h"⟩,⟨.real,[c.BK,c.BV],"ht"⟩,
   ⟨.real,[64,c.BV],"o"⟩,⟨.real,[c.chunks,c.BK,c.BV],"dh"⟩,
   ⟨.real,[64,c.BK],"dq"⟩,⟨.real,[64,c.BK],"dk"⟩,
   ⟨.real,[c.valueBlocks,64,c.BV],"dv"⟩]

abbrev Inputs (α : Type) (c : Shape) := ChannelValues α (inputs c)
abbrev Outputs (α : Type) (c : Shape) := ChannelValues α (outputs c)

def inputEnabled (a : Call) (port : Fin (inputs a.shape).length) : Prop :=
  match port with
  | ⟨0,_⟩ => a.stage=.forwardO ∨ a.stage=.backwardH ∨ a.stage=.backwardQKV
  | ⟨1,_⟩ => a.stage=.forwardH ∨ a.stage=.forwardO ∨ a.stage=.backwardQKV
  | ⟨2,_⟩ => a.stage=.forwardH ∨ a.stage=.forwardO ∨ a.stage=.backwardQKV
  | ⟨3,_⟩ => a.stage=.backwardH ∨ a.stage=.backwardQKV
  | ⟨4,_⟩ => a.stage=.forwardH ∧ a.seed=Bool.true
  | ⟨5,_⟩ => a.stage=.backwardH ∧ a.seed=Bool.true
  | ⟨6,_⟩ => a.stage=.forwardO ∨ a.stage=.backwardQKV
  | ⟨7,_⟩ => a.stage=.backwardQKV

def inputOffset (c : Shape) (port : Fin (inputs c).length) :
    TileIndex (inputs c)[port].shape → Nat := match port with
  | ⟨0,_⟩ | ⟨1,_⟩ => fun j => j.1.val*(c.T*c.K)+j.2.1.val*c.K+j.2.2.1.val
  | ⟨2,_⟩ | ⟨3,_⟩ => fun j => j.1.val*(c.T*c.V)+j.2.1.val*c.V+j.2.2.1.val
  | ⟨4,_⟩ | ⟨5,_⟩ => fun j => j.1.val*c.K*c.V+j.2.1.val*c.V+j.2.2.1.val
  | ⟨6,_⟩ | ⟨7,_⟩ => fun j => j.1.val*(c.chunks*c.K*c.V)+j.2.1.val*c.K*c.V+
      j.2.2.1.val*c.V+j.2.2.2.1.val

def outputOffset (c : Shape) (p : Nat→Nat) (grid : Nat→Nat)
    (port : Fin (outputs c).length) : TileIndex (outputs c)[port].shape → Nat := match port with
  | ⟨0,_⟩ => fun j => p 2*(c.chunks*c.K*c.V)+j.1.val*c.K*c.V+
      (p 0*c.BK+j.2.1.val)*c.V+(p 1*c.BV+j.2.2.1.val)
  | ⟨1,_⟩ => fun j => p 2*c.K*c.V+(p 0*c.BK+j.1.val)*c.V+(p 1*c.BV+j.2.1.val)
  | ⟨2,_⟩ => fun j => p 2*(c.T*c.V)+(p 1*64+j.1.val)*c.V+(p 0*c.BV+j.2.1.val)
  | ⟨3,_⟩ => fun j => p 2*(c.chunks*c.K*c.V)+(c.chunks-1-j.1.val)*c.K*c.V+
      (p 0*c.BK+j.2.1.val)*c.V+(p 1*c.BV+j.2.2.1.val)
  | ⟨4,_⟩ | ⟨5,_⟩ => fun j => p 2*(c.T*c.K)+(p 1*64+j.1.val)*c.K+(p 0*c.BK+j.2.1.val)
  | ⟨6,_⟩ => fun j => (p 0*grid 2+p 2)*(c.T*c.V)+(p 1*64+j.2.1.val)*c.V+
      (j.1.val*c.BV+j.2.2.1.val)

def outputActive (a : Call) (p : Nat→Nat) (port : Fin (outputs a.shape).length) :
    TileIndex (outputs a.shape)[port].shape → Prop := match port with
  | ⟨0,_⟩ => fun j => a.stage=.forwardH ∧ p 0*a.shape.BK+j.2.1.val<a.shape.K ∧
      p 1*a.shape.BV+j.2.2.1.val<a.shape.V
  | ⟨1,_⟩ => fun j => a.stage=.forwardH ∧ a.final=Bool.true ∧
      p 0*a.shape.BK+j.1.val<a.shape.K ∧ p 1*a.shape.BV+j.2.1.val<a.shape.V
  | ⟨2,_⟩ => fun j => a.stage=.forwardO ∧ p 1*64+j.1.val<a.shape.T ∧
      p 0*a.shape.BV+j.2.1.val<a.shape.V
  | ⟨3,_⟩ => fun j => a.stage=.backwardH ∧ p 0*a.shape.BK+j.2.1.val<a.shape.K ∧
      p 1*a.shape.BV+j.2.2.1.val<a.shape.V
  | ⟨4,_⟩ | ⟨5,_⟩ => fun j => a.stage=.backwardQKV ∧ p 1*64+j.1.val<a.shape.T ∧
      p 0*a.shape.BK+j.2.1.val<a.shape.K
  | ⟨6,_⟩ => fun j => a.stage=.backwardQKV ∧ p 1*64+j.2.1.val<a.shape.T ∧
      j.1.val*a.shape.BV+j.2.2.1.val<a.shape.V

/-- Original launch-coordinate domain, independent of execution and values. -/
def Legal (a : Call) (p grid : Nat→Nat) : Prop :=
  p 2<a.shape.B*a.shape.H ∧ grid 2=a.shape.B*a.shape.H ∧
  match a.stage with
  | .forwardH | .backwardH => p 0<a.shape.keyBlocks ∧ p 1<a.shape.valueBlocks
  | .forwardO => p 0<a.shape.valueBlocks ∧ p 1<a.shape.chunks
  | .backwardQKV => p 0<a.shape.keyBlocks ∧ p 1<a.shape.chunks

def launch {α : Type} (M : Algebra α) (a : Call) (p grid : Nat→Nat) : Invocation α where
  kernel := a.kernel
  backend := M
  pids := p
  numPids := some grid

/-- Shared typed memory layout. Source storage/computation stays FP32;
the .real cell carrier is the existing Structural erased representation.
All stage outputs are public, with no private scratch. -/
def layout (α : Type) (a : Call) (p grid : Nat→Nat) : MemoryIO α where
  inputs := inputs a.shape
  outputs := outputs a.shape
  scratch := []
  buffers := ["q","k","v","incoming","h0","terminal","h","dh","ht","o","dq","dk","dv"]
  read := fun port _ => inputOffset a.shape port
  inputMask := fun port _ _ => inputEnabled a port
  write := fun port _ => outputOffset a.shape p grid port
  outputMask := fun port _ => outputActive a p port
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0

def io {α β : Type} (M : Algebra β) (encode : α→β) (observe : β→α)
    (a : Call) (p grid : Nat→Nat) : ProgramIO α β where
  toMemoryIO := layout α a p grid
  schedule := fun _ => [launch M a p grid]
  encode := encode
  observe := fun _ _ => observe

def realIO (a : Call) (p grid : Nat→Nat) : ProgramIO ℝ ℝ :=
  io (FP.ProfiledRewrite.engine (KernelModel.withDot KernelReal.algebra KernelReal.dot)) id id a p grid

/-- The native input representation records only each declared source address.
The source interpreter, source precision and local dot profile are unchanged. -/
def nativeIO {α : Type} (ops : Ops α) (literal : ℝ→α) (a : Call) (p grid : Nat→Nat) :
    AddressedProgramIO α (KernelStaging.Term α) where
  toMemoryIO := layout α a p grid
  schedule := fun _ => [launch
    (FP.ProfiledRewrite.engine (KernelModel.withDot (KernelStaging.algebra ops.zero) (nativeDot a))) a p grid]
  encodeAt := KernelStaging.Term.input
  observe := fun _ _ => KernelStaging.lower ops literal

end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
noncomputable section

def sample3 {α : Type} {a b c : Nat} (zero : α) (xs : TileIndex [a,b,c]→α)
    (i j k : Nat) : α :=
  if hi : i<a then if hj : j<b then if hk : k<c then xs (⟨i,hi⟩,⟨j,hj⟩,⟨k,hk⟩,())
  else zero else zero else zero

def sample4 {α : Type} {a b c d : Nat} (zero : α) (xs : TileIndex [a,b,c,d]→α)
    (i j k l : Nat) : α :=
  if hi : i<a then if hj : j<b then if hk : k<c then if hl : l<d then
    xs (⟨i,hi⟩,⟨j,hj⟩,⟨k,hk⟩,⟨l,hl⟩,()) else zero else zero else zero else zero

/-- Logical arrays for the existing independent retention formulas. Extension
outside a channel's mathematical shape is explicit zero padding; the source
binding proofs must establish every in-range observation used by a formula. -/
def logicalInputs {α : Type} (zero : α) (a : Call) (xs : Inputs α a.shape) :
    Whole001.Inputs α where
  q := sample3 zero (xs 0)
  k := sample3 zero (xs 1)
  v := sample3 zero (xs 2)
  incoming := sample3 zero (xs 3)
  seed := if a.seed then sample3 zero (xs 4) else fun _ _ _ => zero
  finalIncoming := if a.seed then sample3 zero (xs 5) else fun _ _ _ => zero

def history {α : Type} (zero : α) (c : Shape) (xs : Inputs α c)
    (reverse : Bool) (head chunk : Nat) : Tile α :=
  if reverse then sample4 zero (xs 7) head chunk else sample4 zero (xs 6) head chunk

end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.LogicalTensors
open VeriTile Triton FP.Structural KernelPacket
noncomputable section

/- Every finite tensor, including inactive optional channels, has a true
coordinate inverse. The original typed memory/schedule/observer is reused. -/
structure Inputs (c : Shape) where
  query : Fin (c.B*c.H) → Fin (c.T) → Fin (c.K) → ℝ
  key : Fin (c.B*c.H) → Fin (c.T) → Fin (c.K) → ℝ
  value : Fin (c.B*c.H) → Fin (c.T) → Fin (c.V) → ℝ
  incoming : Fin (c.B*c.H) → Fin (c.T) → Fin (c.V) → ℝ
  initial : Fin (c.B*c.H) → Fin (c.K) → Fin (c.V) → ℝ
  terminal : Fin (c.B*c.H) → Fin (c.K) → Fin (c.V) → ℝ
  history : Fin (c.B*c.H) → Fin (c.chunks) → Fin (c.K) → Fin (c.V) → ℝ
  dHistory : Fin (c.B*c.H) → Fin (c.chunks) → Fin (c.K) → Fin (c.V) → ℝ

structure Outputs (c : Shape) where
  history : Fin (c.chunks) → Fin (c.BK) → Fin (c.BV) → ℝ
  finalState : Fin (c.BK) → Fin (c.BV) → ℝ
  output : Fin (64) → Fin (c.BV) → ℝ
  dHistory : Fin (c.chunks) → Fin (c.BK) → Fin (c.BV) → ℝ
  dQuery : Fin (64) → Fin (c.BK) → ℝ
  dKey : Fin (64) → Fin (c.BK) → ℝ
  dValuePartial : Fin (c.valueBlocks) → Fin (64) → Fin (c.BV) → ℝ

def inputs (c : Shape) : PublicIO.Inputs ℝ c ≃ Inputs c where
  toFun := fun values => ⟨Values.curryEquiv ℝ .real [c.B*c.H,c.T,c.K] (values (0 : Fin 8)),
    Values.curryEquiv ℝ .real [c.B*c.H,c.T,c.K] (values (1 : Fin 8)),
    Values.curryEquiv ℝ .real [c.B*c.H,c.T,c.V] (values (2 : Fin 8)),
    Values.curryEquiv ℝ .real [c.B*c.H,c.T,c.V] (values (3 : Fin 8)),
    Values.curryEquiv ℝ .real [c.B*c.H,c.K,c.V] (values (4 : Fin 8)),
    Values.curryEquiv ℝ .real [c.B*c.H,c.K,c.V] (values (5 : Fin 8)),
    Values.curryEquiv ℝ .real [c.B*c.H,c.chunks,c.K,c.V] (values (6 : Fin 8)),
    Values.curryEquiv ℝ .real [c.B*c.H,c.chunks,c.K,c.V] (values (7 : Fin 8))⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (Values.curryEquiv ℝ .real [c.B*c.H,c.T,c.K]).symm values.query
    | ⟨1,_⟩ => (Values.curryEquiv ℝ .real [c.B*c.H,c.T,c.K]).symm values.key
    | ⟨2,_⟩ => (Values.curryEquiv ℝ .real [c.B*c.H,c.T,c.V]).symm values.value
    | ⟨3,_⟩ => (Values.curryEquiv ℝ .real [c.B*c.H,c.T,c.V]).symm values.incoming
    | ⟨4,_⟩ => (Values.curryEquiv ℝ .real [c.B*c.H,c.K,c.V]).symm values.initial
    | ⟨5,_⟩ => (Values.curryEquiv ℝ .real [c.B*c.H,c.K,c.V]).symm values.terminal
    | ⟨6,_⟩ => (Values.curryEquiv ℝ .real [c.B*c.H,c.chunks,c.K,c.V]).symm values.history
    | ⟨7,_⟩ => (Values.curryEquiv ℝ .real [c.B*c.H,c.chunks,c.K,c.V]).symm values.dHistory
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;> exact congrFun (Equiv.symm_apply_apply _ _) j
  right_inv := by
    intro values
    cases values
    dsimp
    congr 1 <;> apply Equiv.apply_symm_apply

def outputs (c : Shape) : PublicIO.Outputs ℝ c ≃ Outputs c where
  toFun := fun values => ⟨Values.curryEquiv ℝ .real [c.chunks,c.BK,c.BV] (values (0 : Fin 7)),
    Values.curryEquiv ℝ .real [c.BK,c.BV] (values (1 : Fin 7)),
    Values.curryEquiv ℝ .real [64,c.BV] (values (2 : Fin 7)),
    Values.curryEquiv ℝ .real [c.chunks,c.BK,c.BV] (values (3 : Fin 7)),
    Values.curryEquiv ℝ .real [64,c.BK] (values (4 : Fin 7)),
    Values.curryEquiv ℝ .real [64,c.BK] (values (5 : Fin 7)),
    Values.curryEquiv ℝ .real [c.valueBlocks,64,c.BV] (values (6 : Fin 7))⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (Values.curryEquiv ℝ .real [c.chunks,c.BK,c.BV]).symm values.history
    | ⟨1,_⟩ => (Values.curryEquiv ℝ .real [c.BK,c.BV]).symm values.finalState
    | ⟨2,_⟩ => (Values.curryEquiv ℝ .real [64,c.BV]).symm values.output
    | ⟨3,_⟩ => (Values.curryEquiv ℝ .real [c.chunks,c.BK,c.BV]).symm values.dHistory
    | ⟨4,_⟩ => (Values.curryEquiv ℝ .real [64,c.BK]).symm values.dQuery
    | ⟨5,_⟩ => (Values.curryEquiv ℝ .real [64,c.BK]).symm values.dKey
    | ⟨6,_⟩ => (Values.curryEquiv ℝ .real [c.valueBlocks,64,c.BV]).symm values.dValuePartial
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;> exact congrFun (Equiv.symm_apply_apply _ _) j
  right_inv := by
    intro values
    cases values
    dsimp
    congr 1 <;> apply Equiv.apply_symm_apply

def logicalInputs (a : Call) (values : Inputs a.shape) :=
  PublicIO.logicalInputs 0 a ((inputs a.shape).symm values)
def history (c : Shape) (values : Inputs c) (future : Bool) (head chunk : Nat) :=
  PublicIO.history 0 c ((inputs c).symm values) future head chunk

def io (a : Call) (p grid : Nat→Nat) : LogicalProgramIO ℝ (ℝ) (Inputs a.shape) (Outputs a.shape) where
  core := PublicIO.realIO a p grid
  inputs := inputs a.shape
  outputs := outputs a.shape

end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.LogicalTensors
