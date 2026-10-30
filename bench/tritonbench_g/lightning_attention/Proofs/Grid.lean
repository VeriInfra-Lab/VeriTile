import bench.tritonbench_g.lightning_attention.Proofs.ForwardProgram
import bench.tritonbench_g.lightning_attention.Proofs.IntraExecution
import bench.tritonbench_g.lightning_attention.Proofs.InterProgram
import bench.tritonbench_g.lightning_attention.Proofs.HostMetadata
import bench.tritonbench_g.chunk_delta_fwd.Sources.Readback
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]
abbrev Stage := Host.Stage

def stageKernel (stage : Stage) (d : Host.Dimensions) : ComputeKernel :=
  match stage with
  | .forward => forward d.parameters
  | .intra => intra d.parameters
  | .inter => inter d.parameters

def grid (stage : Stage) (d : Host.Dimensions) : List PID :=
  (List.range (Host.grid d stage).1).flatMap (fun head =>
    (List.range (Host.grid d stage).2.1).map (fun channel => ⟨head,channel⟩))
def ValidPID (stage : Stage) (d : Host.Dimensions) (pid : PID) :=
  pid.head<(Host.grid d stage).1 ∧ pid.channel<(Host.grid d stage).2.1

theorem mem_grid (stage : Stage) (d : Host.Dimensions) (pid : PID) :
    pid∈grid stage d ↔ ValidPID stage d pid := by
  cases pid
  simp [grid,ValidPID]

def launchState (stage : Stage) (d : Host.Dimensions) (pid : PID) (m : Memory α) : State α :=
  { ChunkDeltaFwd.memoryState m with
    pids := fun n => match n with | 0 => pid.head | 1 => pid.channel | _ => 0
    numPids := fun n => match n with | 0 => (Host.grid d stage).1 | 1 => (Host.grid d stage).2.1 | _ => 1 }

/-- Each program consumes the predecessor's current memory. In particular the
inter stage reads the gradients produced by the complete intra grid. -/
def stageMemory (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (d : Host.Dimensions)
    (pid : PID) (m : Memory α) : Memory α :=
  match stage with
  | .forward => fwdMemory M D d.parameters pid m m
  | .intra => intraMemory M D d.parameters pid.head pid.channel m
  | .inter => interMemory M D d.parameters pid.head m m

theorem stageProgram (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (d : Host.Dimensions)
    (pid : PID) (m : Memory α) :
    ∃ t, FP.Structural.exec (engine (withDot M D)) (stageKernel stage d)
        (launchState stage d pid m)=some t ∧ t.mem=stageMemory M D stage d pid m := by
  cases stage with
  | forward =>
    obtain ⟨t,he,hm,_⟩ := forwardProgram M D d.parameters m (launchState .forward d pid m) (Inputs.refl m)
    exact ⟨t,he,hm⟩
  | intra =>
    obtain ⟨t,he,hm,_⟩ := intraProgram M D d.parameters (launchState .intra d pid m)
    exact ⟨t,he,hm⟩
  | inter =>
    obtain ⟨t,he,hm,_⟩ := interProgram M D d.parameters rfl m (launchState .inter d pid m) (Inputs.refl m)
    exact ⟨t,he,hm⟩

def runGrid (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (d : Host.Dimensions) :
    List PID → Memory α → Option (Memory α)
  | [],m => some m
  | pid::rest,m => do
      let t ← FP.Structural.exec (engine (withDot M D)) (stageKernel stage d) (launchState stage d pid m)
      runGrid M D stage d rest t.mem

def gridMemory (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (d : Host.Dimensions)
    (schedule : List PID) (m : Memory α) : Memory α :=
  schedule.foldl (fun mem pid => stageMemory M D stage d pid mem) m

theorem grid_execution (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (d : Host.Dimensions)
    (schedule : List PID) (m : Memory α) :
    runGrid M D stage d schedule m=some (gridMemory M D stage d schedule m) := by
  induction schedule generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    obtain ⟨t,he,hm⟩ := stageProgram M D stage d pid m
    simp only [runGrid,he,bind,Option.bind_some,hm]
    exact ih _

def executeThree (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions)
    (sf si sb : List PID) (m : Memory α) : Option (Memory α) := do
  let f ← runGrid M D .forward d sf m
  let i ← runGrid M D .intra d si f
  runGrid M D .inter d sb i

def modelThree (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions)
    (sf si sb : List PID) (m : Memory α) : Memory α :=
  gridMemory M D .inter d sb (gridMemory M D .intra d si (gridMemory M D .forward d sf m))

theorem three_execution (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions)
    (sf si sb : List PID) (m : Memory α) :
    executeThree M D d sf si sb m=some (modelThree M D d sf si sb m) := by
  simp only [executeThree,grid_execution,bind,Option.bind_some,modelThree]
#axiomsClean grid_execution
#axiomsClean three_execution
#specNonCircular modelThree avoiding [run,FP.Structural.exec,runGrid,executeThree]
end
end VeriTile.Bench.Repairs.LightningAttention
