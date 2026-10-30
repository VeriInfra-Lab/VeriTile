import bench.tritonbench_g.chunk_retention_ops.Sources.KernelForwardReadback
import bench.tritonbench_g.chunk_retention_ops.Sources.KernelOutputCorrect
import bench.tritonbench_g.chunk_retention_ops.Sources.KernelBackwardFrame
import bench.tritonbench_g.chunk_retention_ops.Sources.KernelGradientFrame
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelPacket
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open KernelBackward (PID Matrix)
open KernelReal (parameters)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (Stage launchState hActive hOffset fOffset scatter_frame)
open VeriTile.Bench.Repairs.ChunkDeltaFwd (foldRange_observe)
noncomputable section
abbrev Memory := KernelBackward.Memory
variable {α : Type} [Inhabited α]

/-- One selected kernel launch. The two flags are its input-state and
output-state switches; no host/autograd/runtime protocol is modeled here. -/
structure Call where
  shape : Shape
  stage : Stage
  scale : ℝ
  inputState : Bool
  outputState : Bool
  pid : PID

def Call.kernel (a : Call) : ComputeKernel := match a.stage with
  | .forwardState=>KernelForward.kernel a.shape a.inputState a.outputState
  | .forwardOutput=>KernelOutput.kernel a.shape a.scale
  | .backwardState=>KernelBackward.kernel (parameters a.shape) a.scale a.inputState a.outputState
  | .backwardGradient=>KernelGradient.kernel (parameters a.shape) a.scale

def Call.state (a : Call) (m : Memory α) : State α := launchState a.stage (parameters a.shape) a.pid m

/-- Input-only recurrence and scatter reference, independent of source syntax
and interpreter execution. Genuine intermediate buffers are ordinary inputs to
the standalone output and QKV kernels. -/
def reference (M : Algebra α) (D : DotPrimitive α) (a : Call) (m : Memory α) : Memory α := match a.stage with
  | .forwardState=>KernelForward.memory M D a.shape a.pid m m a.inputState a.outputState
  | .forwardOutput=>KernelOutput.memory M D a.shape a.scale a.pid m m
  | .backwardState=>KernelBackward.memory M D (parameters a.shape) a.pid m m a.scale a.inputState a.outputState
  | .backwardGradient=>KernelGradient.gradientMemory M D (parameters a.shape) a.pid (a.shape.B*a.shape.H) m m a.scale

theorem source (M : Algebra α) (D : DotPrimitive α) (a : Call) (m : Memory α) :
    ∃ t,FP.Structural.exec (engine (withDot M D)) a.kernel (a.state m)=some t ∧ t.mem=reference M D a m := by
  rcases a with ⟨c,stage,scale,seed,final,pid⟩
  cases stage with
  | forwardState=>exact KernelForward.source M D c seed final pid m
  | forwardOutput=>exact KernelOutput.source M D c scale pid m
  | backwardState=>
    obtain ⟨t,he,hm,_⟩:=KernelBackward.program M D (parameters c) scale seed final m
      (launchState .backwardState (parameters c) pid m) (KernelBackward.InputEq.refl m) (by simp [parameters])
    exact ⟨t,he,hm⟩
  | backwardGradient=>
    exact KernelGradient.gradientProgram M D (parameters c) scale
      (launchState .backwardGradient (parameters c) pid m) (by simp [parameters])

def backwardFootprint (a : Call) (r : RegionName) (o : Nat) :=
  (r="dhistory" ∧ ∃ n,n<a.shape.chunks ∧ ∃ i : TileIndex [a.shape.BK,a.shape.BV],
    hActive (parameters a.shape) a.pid i ∧ o=hOffset (parameters a.shape) a.pid (KernelBackward.time (parameters a.shape) n) i) ∨
  (r="initialGradient" ∧ a.outputState=Bool.true ∧ ∃ i : TileIndex [a.shape.BK,a.shape.BV],
    hActive (parameters a.shape) a.pid i ∧ o=fOffset (parameters a.shape) a.pid i)

def Footprint (a : Call) (r : RegionName) (o : Nat) : Prop := match a.stage with
  | .forwardState=>KernelForward.Footprint a.shape a.pid a.outputState r o
  | .forwardOutput=>KernelOutput.Footprint a.shape a.pid r o
  | .backwardState=>backwardFootprint a r o
  | .backwardGradient=>VeriTile.Bench.Repairs.ChunkLinearAttn.GradientFootprint (parameters a.shape) a.pid r o

theorem backward_frame (M : Algebra α) (D : DotPrimitive α) (a : Call) (m : Memory α)
    (r : RegionName) (o : Nat) (safe : ¬backwardFootprint a r o) :
    KernelBackward.memory M D (parameters a.shape) a.pid m m a.scale a.inputState a.outputState r o=m r o := by
  have hf : ∀ carry mem,KernelBackward.writeInitial M (parameters a.shape) a.pid a.outputState carry mem r o=mem r o := by
    intro carry mem
    cases hflag : a.outputState
    · rfl
    · apply scatter_frame
      by_cases hr : r="initialGradient"
      · right
        intro i _ hi he
        exact safe (Or.inr ⟨hr,hflag,i,hi,he⟩)
      · exact Or.inl hr
  unfold KernelBackward.memory
  rw [hf]
  apply foldRange_observe _ (fun mem : Memory α=>mem r o)
  intro n hn mem
  apply scatter_frame
  by_cases hr : r="dhistory"
  · right
    intro i _ hi he
    exact safe (Or.inl ⟨hr,n,hn,i,hi,he⟩)
  · exact Or.inl hr

theorem frame (M : Algebra α) (D : DotPrimitive α) (a : Call) (m : Memory α)
    (r : RegionName) (o : Nat) (safe : ¬Footprint a r o) : reference M D a m r o=m r o := by
  unfold reference
  cases hstage : a.stage with
  | forwardState=>exact KernelForward.memory_frame M D a.shape a.pid m m a.inputState a.outputState r o (by simpa only [Footprint,hstage] using safe)
  | forwardOutput=>exact KernelOutput.memory_frame M D a.shape a.scale a.pid m m r o (by simpa only [Footprint,hstage] using safe)
  | backwardState=>exact backward_frame M D a m r o (by simpa only [Footprint,hstage] using safe)
  | backwardGradient=>exact KernelGradient.gradient_frame M D (parameters a.shape) a.pid m m a.scale r o (by simpa only [Footprint,hstage] using safe)

/-- Existential completion, result reference, and exact frame are simultaneous
conclusions. Neither successful execution nor a complete result is assumed. -/
theorem complete (M : Algebra α) (D : DotPrimitive α) (a : Call) (m : Memory α) :
    ∃ t,FP.Structural.exec (engine (withDot M D)) a.kernel (a.state m)=some t ∧
      t.mem=reference M D a m ∧ ∀ r off,¬Footprint a r off → t.mem r off=m r off := by
  obtain ⟨t,ht,hm⟩:=source M D a m
  exact ⟨t,ht,hm,fun r o h=>hm ▸ frame M D a m r o h⟩

#axiomsClean source
#axiomsClean frame
#axiomsClean complete
#specNonCircular reference avoiding [run,FP.Structural.exec,source,complete]
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelPacket
