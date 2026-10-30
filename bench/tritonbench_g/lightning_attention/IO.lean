import bench.tritonbench_g.lightning_attention.Grid
import bench.tritonbench_g.lightning_attention.Algebras
import VeriTile.Triton.Float.ProgramIO
import Mathlib.Tactic.FinCases
import VeriTile.Triton.Float.LogicalProgramIO

/-! Public source identity and regional IO. Source loads, arithmetic and stores
remain FP32; the Structural interpreter's floating cell channel is `.real`.
All launch IDs and both backward grids are retained in source order. -/
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section

/-- Original launch ABI. No custom runner or execution premise is installed. -/
def invocation {α : Type} (M : Algebra α) (D : DotPrimitive α)
    (stage : Stage) (d : Host.Dimensions) (pid : PID) : Invocation α where
  kernel := stageKernel stage d
  backend := engine (withDot M D)
  clearRegisters := Bool.true
  pids := fun ax => match ax with | 0 => pid.head | 1 => pid.channel | _ => 0
  numPids := some (fun ax => match ax with
    | 0 => (Host.grid d stage).1 | 1 => (Host.grid d stage).2.1 | _ => 1)

def stageSchedule {α : Type} (M : Algebra α) (D : DotPrimitive α)
    (stage : Stage) (d : Host.Dimensions) : List (Invocation α) :=
  (grid stage d).map (invocation M D stage d)

def inputChannels (d : Host.Dimensions) : List IOChannel :=
  [⟨.real,[d.B*d.H,d.N,d.D],"query"⟩,
   ⟨.real,[d.B*d.H,d.N,d.D],"key"⟩,
   ⟨.real,[d.B*d.H,d.N,d.E],"value"⟩,
   ⟨.real,[d.B*d.H,d.N,d.E],"incoming"⟩]

def forwardChannels (d : Host.Dimensions) : List IOChannel :=
  [⟨.real,[d.B*d.H,d.N,d.E],"output"⟩]

def backwardChannels (d : Host.Dimensions) : List IOChannel :=
  [⟨.real,[d.B*d.H,d.N,d.D],"queryGradient"⟩,
   ⟨.real,[d.B*d.H,d.N,d.D],"keyGradient"⟩,
   ⟨.real,[d.B*d.H,d.N,d.E],"valueGradient"⟩]

/-- Dense head/time/feature indexing of the declared Tensor view. -/
def denseIndex {heads time width : Nat} (i : TileIndex [heads,time,width]) : Nat :=
  i.1.val*time*width+i.2.1.val*width+i.2.2.1.val

def inputIndex (d : Host.Dimensions) (port : Fin (inputChannels d).length) :
    TileIndex ((inputChannels d)[port]).shape → Nat :=
  match port with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ | ⟨3,_⟩ => denseIndex

def backwardIndex (d : Host.Dimensions) (port : Fin (backwardChannels d).length) :
    TileIndex ((backwardChannels d)[port]).shape → Nat :=
  match port with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ => denseIndex

def forwardIO (d : Host.Dimensions) : ProgramIO ℝ ℝ where
  scratch := []
  inputs := inputChannels d
  outputs := forwardChannels d
  buffers := ["query","key","value","output"]
  read := fun i _ => inputIndex d i
  inputMask := fun i _ _ => i.val < 3
  write := fun ⟨0,_⟩ _ => denseIndex
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => stageSchedule Real.algebra Real.dotPrimitive .forward d
  encode := id
  observe := fun _ _ => id

/-- The second launch reads the actual gradients written by the first launch.
These three buffers are public in-place outputs, with no undeclared scratch. -/
def backwardIO (d : Host.Dimensions) : ProgramIO ℝ ℝ where
  scratch := []
  inputs := inputChannels d
  outputs := backwardChannels d
  buffers := ["query","key","value","incoming","queryGradient","keyGradient","valueGradient"]
  read := fun i _ => inputIndex d i
  inputMask := fun _ _ _ => True
  write := fun i _ => backwardIndex d i
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => stageSchedule Real.algebra Real.dotPrimitive .intra d ++
    stageSchedule Real.algebra Real.dotPrimitive .inter d
  encode := id
  observe := fun _ _ => id

abbrev LogicalInputs (d : Host.Dimensions) := ChannelValues ℝ (inputChannels d)
def queryValues (d : Host.Dimensions) (x : LogicalInputs d) (h : Fin (d.B*d.H)) :
    Fin d.N → Fin d.D → ℝ := fun t a => x (0 : Fin 4) (h,t,a,())
def keyValues (d : Host.Dimensions) (x : LogicalInputs d) (h : Fin (d.B*d.H)) :
    Fin d.N → Fin d.D → ℝ := fun t a => x (1 : Fin 4) (h,t,a,())
def valueValues (d : Host.Dimensions) (x : LogicalInputs d) (h : Fin (d.B*d.H)) :
    Fin d.N → Fin d.E → ℝ := fun t a => x (2 : Fin 4) (h,t,a,())
def incomingValues (d : Host.Dimensions) (x : LogicalInputs d) (h : Fin (d.B*d.H)) :
    Fin d.N → Fin d.E → ℝ := fun t a => x (3 : Fin 4) (h,t,a,())


end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.LogicalIO
open VeriTile.Triton FP.Structural
noncomputable section

structure Inputs (d : Host.Dimensions) where
  query : Fin (d.B*d.H) → Fin d.N → Fin d.D → ℝ
  key : Fin (d.B*d.H) → Fin d.N → Fin d.D → ℝ
  value : Fin (d.B*d.H) → Fin d.N → Fin d.E → ℝ
  incoming : Fin (d.B*d.H) → Fin d.N → Fin d.E → ℝ

def inputs (d : Host.Dimensions) : (backwardIO d).Inputs ≃ Inputs d where
  toFun := fun v => ⟨fun i0 i1 i2 => v (0 : Fin 4) (i0,i1,i2,()), fun i0 i1 i2 => v (1 : Fin 4) (i0,i1,i2,()), fun i0 i1 i2 => v (2 : Fin 4) (i0,i1,i2,()), fun i0 i1 i2 => v (3 : Fin 4) (i0,i1,i2,())⟩
  invFun := fun v i => match i with
    | ⟨0,_⟩ => fun j => v.query j.1 j.2.1 j.2.2.1
    | ⟨1,_⟩ => fun j => v.key j.1 j.2.1 j.2.2.1
    | ⟨2,_⟩ => fun j => v.value j.1 j.2.1 j.2.2.1
    | ⟨3,_⟩ => fun j => v.incoming j.1 j.2.1 j.2.2.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> cases j <;> rfl
  right_inv := by
    intro v
    cases v
    rfl

structure Outputs (d : Host.Dimensions) where
  dq : Fin (d.B*d.H) → Fin d.N → Fin d.D → ℝ
  dk : Fin (d.B*d.H) → Fin d.N → Fin d.D → ℝ
  dv : Fin (d.B*d.H) → Fin d.N → Fin d.E → ℝ

def outputs (d : Host.Dimensions) : (backwardIO d).Outputs ≃ Outputs d where
  toFun := fun v => ⟨fun i0 i1 i2 => v (0 : Fin 3) (i0,i1,i2,()), fun i0 i1 i2 => v (1 : Fin 3) (i0,i1,i2,()), fun i0 i1 i2 => v (2 : Fin 3) (i0,i1,i2,())⟩
  invFun := fun v i => match i with
    | ⟨0,_⟩ => fun j => v.dq j.1 j.2.1 j.2.2.1
    | ⟨1,_⟩ => fun j => v.dk j.1 j.2.1 j.2.2.1
    | ⟨2,_⟩ => fun j => v.dv j.1 j.2.1 j.2.2.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> cases j <;> rfl
  right_inv := by
    intro v
    cases v
    rfl

def io (d : Host.Dimensions) : LogicalProgramIO ℝ ℝ (Inputs d) (Outputs d) where
  core := backwardIO d
  inputs := inputs d
  outputs := outputs d


def forwardOutputs (d : Host.Dimensions) : (forwardIO d).Outputs ≃
    (Fin (d.B*d.H) → Fin d.N → Fin d.E → ℝ) where
  toFun := fun v h t c => v (0 : Fin 1) (h,t,c,())
  invFun := fun v ⟨0,_⟩ j => v j.1 j.2.1 j.2.2.1
  left_inv := by
    intro v
    funext i j
    fin_cases i
    rfl
  right_inv := fun _ => rfl

def forward (d : Host.Dimensions) : LogicalProgramIO ℝ ℝ (Inputs d)
    (Fin (d.B*d.H) → Fin d.N → Fin d.E → ℝ) where
  core := forwardIO d
  inputs := inputs d
  outputs := forwardOutputs d

end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.LogicalIO
