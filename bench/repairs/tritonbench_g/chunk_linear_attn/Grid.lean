import bench.repairs.tritonbench_g.chunk_linear_attn.StateProgram
import bench.repairs.tritonbench_g.chunk_linear_attn.OutputProgram
import bench.repairs.tritonbench_g.chunk_linear_attn.GradientProgram
import bench.repairs.tritonbench_g.chunk_linear_attn.Ownership
import bench.repairs.tritonbench_g.chunk_delta_fwd.Readback
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]
inductive Stage where
  | forwardState | forwardOutput | backwardState | backwardGradient
  deriving DecidableEq

def stageKernel (stage : Stage) (p : Parameters) (scale : ℝ) (seed final : Bool) : ComputeKernel :=
  match stage with
  | .forwardState => forwardH p seed final
  | .forwardOutput => forwardO p scale
  | .backwardState => backwardH p scale seed final
  | .backwardGradient => backwardQKV p scale

def gridSize (stage : Stage) (p : Parameters) : PID :=
  match stage with
  | .forwardState | .backwardState => ⟨keyBlocks p,valueBlocks p,p.B*p.H⟩
  | .forwardOutput => ⟨valueBlocks p,p.NT,p.B*p.H⟩
  | .backwardGradient => ⟨keyBlocks p,p.NT,p.B*p.H⟩
def grid (stage : Stage) (p : Parameters) : List PID :=
  (List.range (gridSize stage p).x).flatMap (fun x =>
    (List.range (gridSize stage p).y).flatMap (fun y =>
      (List.range (gridSize stage p).z).map (fun z => ⟨x,y,z⟩)))
def ValidPID (stage : Stage) (p : Parameters) (pid : PID) :=
  pid.x < (gridSize stage p).x ∧ pid.y < (gridSize stage p).y ∧ pid.z < (gridSize stage p).z

theorem mem_grid (stage : Stage) (p : Parameters) (pid : PID) : pid∈grid stage p ↔ ValidPID stage p pid := by
  cases pid
  simp [grid,ValidPID]

def launchState (stage : Stage) (p : Parameters) (pid : PID) (m : Memory α) : State α :=
  { VeriTile.Bench.Repairs.ChunkDeltaFwd.memoryState m with
    pids := fun n => match n with | 0 => pid.x | 1 => pid.y | _ => pid.z
    numPids := fun n => match n with | 0 => (gridSize stage p).x | 1 => (gridSize stage p).y | _ => (gridSize stage p).z }

/-- Each program model consumes its actual current memory. There is no
immutable grid-entry input substituted for a predecessor's writes. -/
def stageMemory (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (pid : PID) (m : Memory α) : Memory α :=
  match stage with
  | .forwardState => stateMemory M D p pid m m scale Bool.false seed final
  | .forwardOutput => outputMemory M D p pid m m scale
  | .backwardState => stateMemory M D p pid m m scale Bool.true seed final
  | .backwardGradient => gradientMemory M D p pid (p.B*p.H) m m scale

theorem stageProgram (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (pid : PID) (m : Memory α) :
    ∃ t, FP.Structural.exec (engine (withDot M D)) (stageKernel stage p scale seed final)
        (launchState stage p pid m)=some t ∧
      t.mem=stageMemory M D stage p scale seed final pid m := by
  cases stage with
  | forwardState =>
    obtain ⟨t,he,hm,_⟩ := stateProgram M D p scale Bool.false seed final m
      (launchState .forwardState p pid m) (StateInputs.refl m)
    exact ⟨t,he,hm⟩
  | backwardState =>
    obtain ⟨t,he,hm,_⟩ := stateProgram M D p scale Bool.true seed final m
      (launchState .backwardState p pid m) (StateInputs.refl m)
    exact ⟨t,he,hm⟩
  | forwardOutput => exact outputProgram M D p scale (launchState .forwardOutput p pid m)
  | backwardGradient => exact gradientProgram M D p scale (launchState .backwardGradient p pid m)

def runGrid (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) : List PID → Memory α → Option (Memory α)
  | [],m => some m
  | pid::rest,m => do
      let t ← FP.Structural.exec (engine (withDot M D)) (stageKernel stage p scale seed final)
        (launchState stage p pid m)
      runGrid M D stage p scale seed final rest t.mem

def gridMemory (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (schedule : List PID) (m : Memory α) : Memory α :=
  schedule.foldl (fun mem pid => stageMemory M D stage p scale seed final pid mem) m

theorem grid_execution (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (schedule : List PID) (m : Memory α) :
    runGrid M D stage p scale seed final schedule m=
      some (gridMemory M D stage p scale seed final schedule m) := by
  induction schedule generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    obtain ⟨t,he,hm⟩ := stageProgram M D stage p scale seed final pid m
    simp only [runGrid,he,bind,Option.bind_some,hm]
    exact ih _

def executeFour (M : Algebra α) (D : DotPrimitive α) (forward backward : Parameters)
    (scale : ℝ) (seed final terminal : Bool) (sf so sb sg : List PID) (m : Memory α) : Option (Memory α) := do
  let hf ← runGrid M D .forwardState forward scale seed final sf m
  let of ← runGrid M D .forwardOutput forward scale Bool.false Bool.false so hf
  let hb ← runGrid M D .backwardState backward scale terminal seed sb of
  runGrid M D .backwardGradient backward scale Bool.false Bool.false sg hb

def modelFour (M : Algebra α) (D : DotPrimitive α) (forward backward : Parameters)
    (scale : ℝ) (seed final terminal : Bool) (sf so sb sg : List PID) (m : Memory α) : Memory α :=
  let hf := gridMemory M D .forwardState forward scale seed final sf m
  let of := gridMemory M D .forwardOutput forward scale Bool.false Bool.false so hf
  let hb := gridMemory M D .backwardState backward scale terminal seed sb of
  gridMemory M D .backwardGradient backward scale Bool.false Bool.false sg hb

theorem four_execution (M : Algebra α) (D : DotPrimitive α) (forward backward : Parameters)
    (scale : ℝ) (seed final terminal : Bool) (sf so sb sg : List PID) (m : Memory α) :
    executeFour M D forward backward scale seed final terminal sf so sb sg m=
      some (modelFour M D forward backward scale seed final terminal sf so sb sg m) := by
  simp only [executeFour,grid_execution,bind,Option.bind_some,modelFour]

#axiomsClean stageProgram
#axiomsClean grid_execution
#axiomsClean four_execution
#specNonCircular modelFour avoiding [run,FP.Structural.exec,runGrid,executeFour]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
