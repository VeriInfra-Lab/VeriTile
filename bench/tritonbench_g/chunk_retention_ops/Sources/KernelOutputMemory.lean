import bench.tritonbench_g.chunk_retention_ops.Sources.KernelForwardO
import bench.tritonbench_g.chunk_retention_ops.Sources.KernelRealModel
import bench.tritonbench_g.chunk_linear_attn.Grid
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOutput
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open KernelBackward (Parameters PID Matrix)
abbrev Memory := KernelBackward.Memory
open KernelReal (parameters)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (scatter scatter_frame outputActive outputOffset)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
namespace S
export VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource
  (value written accumulated update queryGate featureLoad keyLoad historyLoad valueLoad attention)
end S
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def sharedDot (D : DotPrimitive α) : Shared.DotPrimitive α := ⟨D.apply⟩
theorem withDot_eq (M : Algebra α) (D : DotPrimitive α) :
    Shared.withDot M (sharedDot D)=withDot M D := rfl

def state (c : Shape) (pid : PID) (m : Memory α) : State α :=
  VeriTile.Bench.Repairs.ChunkLinearAttn.launchState .forwardOutput (parameters c) pid m

def outputValue (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (pid : PID) (input : Memory α) : Matrix α 64 c.BV :=
  S.value M (sharedDot D) (shape c) scale (state c pid input)
def memory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (pid : PID) (input current : Memory α) : Memory α :=
  scatter (TileShape.allIndices [64,c.BV]) "o" (outputActive (parameters c) pid)
    (outputOffset (parameters c) pid) (fun i=>.mk .real (outputValue M D c scale pid input i)) current

theorem written_memory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (pid : PID) (m : Memory α) :
    (S.written M (sharedDot D) (shape c) scale (state c pid m)).mem=memory M D c scale pid m m := by
  simp [S.written,memory,scatter,masked_scatter_memory,S.value,
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.active,
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.outputAddress,
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.row,
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.column,
    outputValue,outputActive,outputOffset,state,parameters,shape,
    VeriTile.Bench.Repairs.ChunkLinearAttn.launchState]
  rfl

theorem source (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (pid : PID) (m : Memory α) :
    ∃ t,FP.Structural.exec (engine (withDot M D)) (kernel c scale) (state c pid m)=some t ∧
      t.mem=memory M D c scale pid m m := by
  obtain ⟨t,ht,hm⟩:=program M (sharedDot D) c scale (state c pid m)
  rw [withDot_eq] at ht
  exact ⟨t,ht,hm.trans (written_memory M D c scale pid m)⟩

def InputEq (a b : Memory α) : Prop :=
  ∀ r : RegionName,r∈(["q","k","v","h"] : List RegionName) → ∀ o,a r o=b r o

theorem outputValue_inputs (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (pid : PID) (a b : Memory α) (h : InputEq a b) :
    outputValue M D c scale pid a=outputValue M D c scale pid b := by
  have hq:=h "q" (by simp)
  have hk:=h "k" (by simp)
  have hv:=h "v" (by simp)
  have hh:=h "h" (by simp)
  have hu : S.update M (sharedDot D) (shape c) (state c pid a)=
      S.update M (sharedDot D) (shape c) (state c pid b) := by
    funext f acc
    unfold S.update S.queryGate S.featureLoad S.keyLoad S.historyLoad
    unfold VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.rowDecay
    simp [state,VeriTile.Bench.Repairs.ChunkLinearAttn.launchState,
      VeriTile.Bench.Repairs.ChunkDeltaFwd.memoryState,hq,hk,hh,
      VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.row,
      VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.column]
  have ha : S.accumulated M (sharedDot D) (shape c) (state c pid a)=
      S.accumulated M (sharedDot D) (shape c) (state c pid b) := by
    unfold S.accumulated
    rw [hu]
  funext i
  unfold outputValue S.value S.attention
  rw [ha]
  unfold S.valueLoad
  unfold VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.scoreDecay
  simp [state,VeriTile.Bench.Repairs.ChunkLinearAttn.launchState,
    VeriTile.Bench.Repairs.ChunkDeltaFwd.memoryState,hv,
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.active,
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.outputAddress,
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.row,
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.column]

#axiomsClean source
#axiomsClean outputValue_inputs
#specNonCircular memory avoiding [run,FP.Structural.exec,program,source]
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOutput
