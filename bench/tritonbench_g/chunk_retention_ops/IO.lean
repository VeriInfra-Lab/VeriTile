import bench.tritonbench_g.chunk_retention_ops.Proofs.KernelNative
import VeriTile.Triton.Float.AddressedInputsIO
import VeriTile.Triton.Float.LogicalProgramIO

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
open KernelBackward (PID)
open VeriTile.Bench.Repairs.ChunkLinearAttn (Stage gridSize)
noncomputable section

/-- Finite head-local inputs of one standalone stage. The base head is the
actual program coordinate, so the interface does not impose a new head bound.
Saved histories are input slices at the actual selected chunk. -/
@[reducible] def inputs (c : Shape) : List IOChannel :=
  [⟨.real,[c.T,c.K],"q"⟩,⟨.real,[c.T,c.K],"k"⟩,
   ⟨.real,[c.T,c.V],"v"⟩,⟨.real,[c.T,c.V],"incoming"⟩,
   ⟨.real,[c.K,c.V],"h0"⟩,⟨.real,[c.K,c.V],"terminal"⟩,
   ⟨.real,[c.K,c.V],"h"⟩,⟨.real,[c.K,c.V],"dhistory"⟩]

/-- All eight output families, including the optional initial-state gradient
and every per-key-block partial dV. No host pipeline or final dV reduction is
substituted for the four selected standalone kernels. -/
@[reducible] def outputs (c : Shape) : List IOChannel :=
  [⟨.real,[c.chunks,c.BK,c.BV],"h"⟩,⟨.real,[c.BK,c.BV],"ht"⟩,
   ⟨.real,[64,c.BV],"o"⟩,⟨.real,[c.chunks,c.BK,c.BV],"dhistory"⟩,
   ⟨.real,[c.BK,c.BV],"initialGradient"⟩,
   ⟨.real,[64,c.BK],"queryGradient"⟩,⟨.real,[64,c.BK],"keyGradient"⟩,
   ⟨.real,[c.valueBlocks,64,c.BV],"valuePartials"⟩]

abbrev Inputs (α : Type) (c : Shape) := ChannelValues α (inputs c)
abbrev Outputs (α : Type) (c : Shape) := ChannelValues α (outputs c)

def inputEnabled (a : Call) (port : Fin (inputs a.shape).length) : Prop :=
  match port with
  | ⟨0,_⟩ => a.stage=.forwardOutput ∨ a.stage=.backwardState ∨ a.stage=.backwardGradient
  | ⟨1,_⟩ => a.stage=.forwardState ∨ a.stage=.forwardOutput ∨ a.stage=.backwardGradient
  | ⟨2,_⟩ => a.stage=.forwardState ∨ a.stage=.forwardOutput ∨ a.stage=.backwardGradient
  | ⟨3,_⟩ => a.stage=.backwardState ∨ a.stage=.backwardGradient
  | ⟨4,_⟩ => a.stage=.forwardState ∧ a.inputState=Bool.true
  | ⟨5,_⟩ => a.stage=.backwardState ∧ a.inputState=Bool.true
  | ⟨6,_⟩ => a.stage=.forwardOutput ∨ a.stage=.backwardGradient
  | ⟨7,_⟩ => a.stage=.backwardGradient

def inputOffset (a : Call) (port : Fin (inputs a.shape).length) :
    TileIndex (inputs a.shape)[port].shape → Nat := match port with
  | ⟨0,_⟩ | ⟨1,_⟩ => fun j => a.pid.z*(a.shape.T*a.shape.K)+j.1.val*a.shape.K+j.2.1.val
  | ⟨2,_⟩ | ⟨3,_⟩ => fun j => a.pid.z*(a.shape.T*a.shape.V)+j.1.val*a.shape.V+j.2.1.val
  | ⟨4,_⟩ | ⟨5,_⟩ => fun j => a.pid.z*a.shape.K*a.shape.V+j.1.val*a.shape.V+j.2.1.val
  | ⟨6,_⟩ | ⟨7,_⟩ => fun j => a.pid.z*(a.shape.chunks*a.shape.K*a.shape.V)+
      a.pid.y*a.shape.K*a.shape.V+j.1.val*a.shape.V+j.2.1.val

def outputOffset (a : Call) (port : Fin (outputs a.shape).length) :
    TileIndex (outputs a.shape)[port].shape → Nat := match port with
  | ⟨0,_⟩ => fun j => a.pid.z*(a.shape.chunks*a.shape.K*a.shape.V)+j.1.val*a.shape.K*a.shape.V+
      (a.pid.x*a.shape.BK+j.2.1.val)*a.shape.V+(a.pid.y*a.shape.BV+j.2.2.1.val)
  | ⟨1,_⟩ | ⟨4,_⟩ => fun j => a.pid.z*a.shape.K*a.shape.V+
      (a.pid.x*a.shape.BK+j.1.val)*a.shape.V+(a.pid.y*a.shape.BV+j.2.1.val)
  | ⟨2,_⟩ => fun j => a.pid.z*(a.shape.T*a.shape.V)+(a.pid.y*64+j.1.val)*a.shape.V+
      (a.pid.x*a.shape.BV+j.2.1.val)
  | ⟨3,_⟩ => fun j => a.pid.z*(a.shape.chunks*a.shape.K*a.shape.V)+
      (a.shape.chunks-1-j.1.val)*a.shape.K*a.shape.V+
      (a.pid.x*a.shape.BK+j.2.1.val)*a.shape.V+(a.pid.y*a.shape.BV+j.2.2.1.val)
  | ⟨5,_⟩ | ⟨6,_⟩ => fun j => a.pid.z*(a.shape.T*a.shape.K)+(a.pid.y*64+j.1.val)*a.shape.K+
      (a.pid.x*a.shape.BK+j.2.1.val)
  | ⟨7,_⟩ => fun j => (a.pid.x*(a.shape.B*a.shape.H)+a.pid.z)*(a.shape.T*a.shape.V)+
      (a.pid.y*64+j.2.1.val)*a.shape.V+(j.1.val*a.shape.BV+j.2.2.1.val)

def outputActive (a : Call) (port : Fin (outputs a.shape).length) :
    TileIndex (outputs a.shape)[port].shape → Prop := match port with
  | ⟨0,_⟩ => fun j => a.stage=.forwardState ∧ a.pid.x*a.shape.BK+j.2.1.val<a.shape.K ∧
      a.pid.y*a.shape.BV+j.2.2.1.val<a.shape.V
  | ⟨1,_⟩ => fun j => a.stage=.forwardState ∧ a.outputState=Bool.true ∧
      a.pid.x*a.shape.BK+j.1.val<a.shape.K ∧ a.pid.y*a.shape.BV+j.2.1.val<a.shape.V
  | ⟨2,_⟩ => fun j => a.stage=.forwardOutput ∧ a.pid.y*64+j.1.val<a.shape.T ∧
      a.pid.x*a.shape.BV+j.2.1.val<a.shape.V
  | ⟨3,_⟩ => fun j => a.stage=.backwardState ∧ a.pid.x*a.shape.BK+j.2.1.val<a.shape.K ∧
      a.pid.y*a.shape.BV+j.2.2.1.val<a.shape.V
  | ⟨4,_⟩ => fun j => a.stage=.backwardState ∧ a.outputState=Bool.true ∧
      a.pid.x*a.shape.BK+j.1.val<a.shape.K ∧ a.pid.y*a.shape.BV+j.2.1.val<a.shape.V
  | ⟨5,_⟩ | ⟨6,_⟩ => fun j => a.stage=.backwardGradient ∧ a.pid.y*64+j.1.val<a.shape.T ∧
      a.pid.x*a.shape.BK+j.2.1.val<a.shape.K
  | ⟨7,_⟩ => fun j => a.stage=.backwardGradient ∧ a.pid.y*64+j.2.1.val<a.shape.T ∧
      j.1.val*a.shape.BV+j.2.2.1.val<a.shape.V

/-- The actual source body, fixed program coordinates, and original grid axes.
The body supplies all its local registers; no tensor-runtime execution is added. -/
def launch {α : Type} (M : Algebra α) (a : Call) : Invocation α where
  kernel := a.kernel
  backend := M
  pids := fun ax => match ax with | 0=>a.pid.x | 1=>a.pid.y | _=>a.pid.z
  numPids := some (fun ax => match ax with
    | 0=>(gridSize a.stage (KernelReal.parameters a.shape)).x
    | 1=>(gridSize a.stage (KernelReal.parameters a.shape)).y
    | _=>(gridSize a.stage (KernelReal.parameters a.shape)).z)

/-- Source storage and computation remain FP32. The .real channel tags are
the existing Structural erased carrier; they do not replace native FP32 by
exact arithmetic. All written buffers are public and private scratch is empty. -/
def layout (α : Type) (a : Call) : MemoryIO α where
  inputs := inputs a.shape
  outputs := outputs a.shape
  scratch := []
  buffers := ["q","k","v","incoming","h0","terminal","h","dhistory",
    "ht","o","initialGradient","queryGradient","keyGradient","valuePartials"]
  read := fun port _ => inputOffset a port
  inputMask := fun port _ _ => inputEnabled a port
  write := fun port _ => outputOffset a port
  outputMask := fun port _ => outputActive a port
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0

def io {α β : Type} (M : Algebra β) (encode : α→β) (observe : β→α) (a : Call) :
    ProgramIO α β where
  toMemoryIO := layout α a
  schedule := fun _ => [launch M a]
  encode := encode
  observe := fun _ _ => observe

def realIO (a : Call) : ProgramIO ℝ ℝ :=
  io (FP.ProfiledRewrite.engine
    (VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput.withDot KernelReal.algebra KernelReal.dotPrimitive)) id id a

/-- The original native address-labelled input model and stage-specific dot
profile. Scalar ISA observation and literal encoding remain explicit. -/
def nativeIO (isa : ISA) (literal : ℝ→KernelNative.Word) (a : Call) :
    AddressedProgramIO KernelNative.Word KernelNative.Term where
  toMemoryIO := layout KernelNative.Word a
  schedule := fun _ => [launch (FP.ProfiledRewrite.engine
    (VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput.withDot KernelNative.model (KernelNative.dotModel a))) a]
  encodeAt := VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelStaging.Term.input
  observe := fun _ _ => KernelNative.S.lower (KernelNative.sharedOps (wordOps isa)) literal

end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
noncomputable section

/-- Zero extension is used only to give independent scalar formulas total
arguments. Active coordinates always select the declared finite input. -/
def sample2 {α : Type} {rows cols : Nat} (zero : α)
    (v : TileIndex [rows,cols]→α) (r c : Nat) : α :=
  if hr : r<rows then if hc : c<cols then v (⟨r,hr⟩,⟨c,hc⟩,()) else zero else zero

/-- Logical inputs of the selected head. All formulas below evaluate one head;
head-dependent decay still uses the actual head number in the call. -/
def logicalInputs {α : Type} (zero : α) (a : Call) (xs : Inputs α a.shape) :
    Whole001.Inputs α where
  q := fun _ => sample2 zero (xs 0)
  k := fun _ => sample2 zero (xs 1)
  v := fun _ => sample2 zero (xs 2)
  incoming := fun _ => sample2 zero (xs 3)
  seed := fun _ e j => if a.inputState then sample2 zero (xs 4) e j else zero
  finalIncoming := fun _ e j => if a.inputState then sample2 zero (xs 5) e j else zero

def history {α : Type} (zero : α) (c : Shape) (xs : Inputs α c) (reverse : Bool) : Tile α :=
  if reverse then sample2 zero (xs 7) else sample2 zero (xs 6)

end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.LogicalTensors
open VeriTile Triton FP.Structural KernelPacket
noncomputable section

/- Every finite tensor, including inactive optional channels, has a true
coordinate inverse. The original typed memory/schedule/observer is reused. -/
structure Inputs (c : Shape) where
  query : Fin (c.T) → Fin (c.K) → ℝ
  key : Fin (c.T) → Fin (c.K) → ℝ
  value : Fin (c.T) → Fin (c.V) → ℝ
  incoming : Fin (c.T) → Fin (c.V) → ℝ
  initial : Fin (c.K) → Fin (c.V) → ℝ
  terminal : Fin (c.K) → Fin (c.V) → ℝ
  history : Fin (c.K) → Fin (c.V) → ℝ
  dHistory : Fin (c.K) → Fin (c.V) → ℝ

structure Outputs (c : Shape) where
  history : Fin (c.chunks) → Fin (c.BK) → Fin (c.BV) → ℝ
  finalState : Fin (c.BK) → Fin (c.BV) → ℝ
  output : Fin (64) → Fin (c.BV) → ℝ
  dHistory : Fin (c.chunks) → Fin (c.BK) → Fin (c.BV) → ℝ
  dInitial : Fin (c.BK) → Fin (c.BV) → ℝ
  dQuery : Fin (64) → Fin (c.BK) → ℝ
  dKey : Fin (64) → Fin (c.BK) → ℝ
  dValuePartial : Fin (c.valueBlocks) → Fin (64) → Fin (c.BV) → ℝ

def inputs (c : Shape) : PublicIO.Inputs ℝ c ≃ Inputs c where
  toFun := fun values => ⟨Values.curryEquiv ℝ .real [c.T,c.K] (values (0 : Fin 8)),
    Values.curryEquiv ℝ .real [c.T,c.K] (values (1 : Fin 8)),
    Values.curryEquiv ℝ .real [c.T,c.V] (values (2 : Fin 8)),
    Values.curryEquiv ℝ .real [c.T,c.V] (values (3 : Fin 8)),
    Values.curryEquiv ℝ .real [c.K,c.V] (values (4 : Fin 8)),
    Values.curryEquiv ℝ .real [c.K,c.V] (values (5 : Fin 8)),
    Values.curryEquiv ℝ .real [c.K,c.V] (values (6 : Fin 8)),
    Values.curryEquiv ℝ .real [c.K,c.V] (values (7 : Fin 8))⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (Values.curryEquiv ℝ .real [c.T,c.K]).symm values.query
    | ⟨1,_⟩ => (Values.curryEquiv ℝ .real [c.T,c.K]).symm values.key
    | ⟨2,_⟩ => (Values.curryEquiv ℝ .real [c.T,c.V]).symm values.value
    | ⟨3,_⟩ => (Values.curryEquiv ℝ .real [c.T,c.V]).symm values.incoming
    | ⟨4,_⟩ => (Values.curryEquiv ℝ .real [c.K,c.V]).symm values.initial
    | ⟨5,_⟩ => (Values.curryEquiv ℝ .real [c.K,c.V]).symm values.terminal
    | ⟨6,_⟩ => (Values.curryEquiv ℝ .real [c.K,c.V]).symm values.history
    | ⟨7,_⟩ => (Values.curryEquiv ℝ .real [c.K,c.V]).symm values.dHistory
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
  toFun := fun values => ⟨Values.curryEquiv ℝ .real [c.chunks,c.BK,c.BV] (values (0 : Fin 8)),
    Values.curryEquiv ℝ .real [c.BK,c.BV] (values (1 : Fin 8)),
    Values.curryEquiv ℝ .real [64,c.BV] (values (2 : Fin 8)),
    Values.curryEquiv ℝ .real [c.chunks,c.BK,c.BV] (values (3 : Fin 8)),
    Values.curryEquiv ℝ .real [c.BK,c.BV] (values (4 : Fin 8)),
    Values.curryEquiv ℝ .real [64,c.BK] (values (5 : Fin 8)),
    Values.curryEquiv ℝ .real [64,c.BK] (values (6 : Fin 8)),
    Values.curryEquiv ℝ .real [c.valueBlocks,64,c.BV] (values (7 : Fin 8))⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (Values.curryEquiv ℝ .real [c.chunks,c.BK,c.BV]).symm values.history
    | ⟨1,_⟩ => (Values.curryEquiv ℝ .real [c.BK,c.BV]).symm values.finalState
    | ⟨2,_⟩ => (Values.curryEquiv ℝ .real [64,c.BV]).symm values.output
    | ⟨3,_⟩ => (Values.curryEquiv ℝ .real [c.chunks,c.BK,c.BV]).symm values.dHistory
    | ⟨4,_⟩ => (Values.curryEquiv ℝ .real [c.BK,c.BV]).symm values.dInitial
    | ⟨5,_⟩ => (Values.curryEquiv ℝ .real [64,c.BK]).symm values.dQuery
    | ⟨6,_⟩ => (Values.curryEquiv ℝ .real [64,c.BK]).symm values.dKey
    | ⟨7,_⟩ => (Values.curryEquiv ℝ .real [c.valueBlocks,64,c.BV]).symm values.dValuePartial
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
def history (c : Shape) (values : Inputs c) (future : Bool) :=
  PublicIO.history 0 c ((inputs c).symm values) future

def io (a : Call) : LogicalProgramIO ℝ (ℝ) (Inputs a.shape) (Outputs a.shape) where
  core := PublicIO.realIO a
  inputs := inputs a.shape
  outputs := outputs a.shape

end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.LogicalTensors
