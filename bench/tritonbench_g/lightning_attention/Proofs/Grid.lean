import bench.tritonbench_g.lightning_attention.Kernels
import bench.tritonbench_g.lightning_attention.Proofs.ForwardBody
import bench.tritonbench_g.lightning_attention.Proofs.IntraModel
import bench.tritonbench_g.lightning_attention.Proofs.InterReverseProgram
import bench.tritonbench_g.lightning_attention.Proofs.HostMetadata
import bench.tritonbench_g.chunk_delta_fwd.Sources.Readback
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]
abbrev Stage := Host.Stage

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

def gridMemory (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (d : Host.Dimensions)
    (schedule : List PID) (m : Memory α) : Memory α :=
  schedule.foldl (fun mem pid => stageMemory M D stage d pid mem) m

def modelThree (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions)
    (sf si sb : List PID) (m : Memory α) : Memory α :=
  gridMemory M D .inter d sb (gridMemory M D .intra d si (gridMemory M D .forward d sf m))

#specNonCircular modelThree avoiding [VeriTile.Bench.Repairs.LightningAttention.KernelOnly.forward,run,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.LightningAttention
