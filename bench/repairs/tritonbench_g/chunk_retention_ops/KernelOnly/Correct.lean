import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.Correct
open VeriTile Triton FP.Structural KernelPacket
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
open KernelBackward (PID)
open scoped BigOperators
noncomputable section

/- Independent scalar mathematics and the complete logical output channels.
These definitions are definitionally connected to the canonical target used below. -/
namespace Mathematics
open PublicIO (Inputs)
def headRate (c : Shape) (head : Nat) : ℝ :=
  1-Real.exp ((-5-(head%c.H : Nat) : ℝ)*Real.log 2)

def weighted (rate seed : ℝ) (term : Nat → ℝ) (time : Nat) : ℝ :=
  rate^time*seed+∑ u ∈ Finset.range time, rate^(time-1-u)*term u

def suffix (rate terminal : ℝ) (term : Nat → ℝ) (start count : Nat) : ℝ :=
  rate^count*terminal + ∑ i ∈ Finset.range count, rate^(i+1)*term (start+i)

def state (c : Shape) (xs : Inputs ℝ) (hasSeed : Bool)
    (head time key value : Nat) : ℝ :=
  weighted (headRate c head) (if hasSeed then xs.seed head key value else 0)
    (fun u => xs.k head u key*xs.v head u value) time

def forcing (xs : Inputs ℝ) (scale : ℝ) (head key value time : Nat) :=
  scale*xs.q head time key*xs.incoming head time value

def futureState (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head time key value : Nat) :=
  suffix (headRate c head) (xs.finalIncoming head key value)
    (forcing xs scale head key value) time (c.T-time)

def finalState (c : Shape) (xs : Inputs ℝ) (hasSeed : Bool)
    (head key value : Nat) : ℝ := state c xs hasSeed head c.T key value

def initialGradient (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head key value : Nat) : ℝ := futureState c xs scale head 0 key value

def localCoefficient (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head chunk lane key value : Nat) (dhistory : Nat → Nat → ℝ) : ℝ :=
  (headRate c head)^(c.length chunk-1-lane)*dhistory key value+
  ∑ i ∈ Finset.range (c.length chunk-lane), (headRate c head)^i*
    forcing xs scale head key value (chunk*64+lane+i)

def outputTile (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head chunk lane value : Nat)
    (history : Nat → Nat → ℝ) : ℝ :=
  ((∑ key ∈ Finset.range c.K,
      (xs.q head (chunk*64+lane) key*(headRate c head)^(lane+1))*history key value)+
    ∑ i ∈ Finset.range (lane+1),
      (∑ key ∈ Finset.range c.K, xs.q head (chunk*64+lane) key*xs.k head (chunk*64+i) key)*
      ((headRate c head)^(lane-i)*xs.v head (chunk*64+i) value))*scale

def queryTile (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head chunk lane key : Nat)
    (history : Nat → Nat → ℝ) : ℝ :=
  ((∑ value ∈ Finset.range c.V,
      (xs.incoming head (chunk*64+lane) value*(headRate c head)^(lane+1))*history key value)+
    ∑ i ∈ Finset.range (lane+1),
      (∑ value ∈ Finset.range c.V,
        xs.incoming head (chunk*64+lane) value*xs.v head (chunk*64+i) value)*
      ((headRate c head)^(lane-i)*xs.k head (chunk*64+i) key))*scale

def keyTile (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head chunk lane key : Nat)
    (dhistory : Nat → Nat → ℝ) : ℝ :=
  (∑ value ∈ Finset.range c.V, xs.v head (chunk*64+lane) value*dhistory key value)*
    (headRate c head)^(c.length chunk-1-lane)+
  ∑ i ∈ Finset.range (c.length chunk-lane),
    (∑ value ∈ Finset.range c.V,
      xs.v head (chunk*64+lane) value*xs.incoming head (chunk*64+lane+i) value)*
    ((headRate c head)^i*scale*xs.q head (chunk*64+lane+i) key)

def valuePartial (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (block head chunk lane value : Nat) (dhistory : Nat → Nat → ℝ) : ℝ :=
  ∑ i ∈ Finset.range c.BK, if block*c.BK+i<c.K then
    xs.k head (chunk*64+lane) (block*c.BK+i)*
      localCoefficient c xs scale head chunk lane (block*c.BK+i) value dhistory else 0

def mathOutputs (a : KernelPacket.Call) (data : Whole001.Inputs ℝ) (saved future : Tile ℝ) : LogicalTensors.Outputs a.shape :=
  let c:=a.shape
  { history := fun chunk key value => state c data Bool.true a.pid.z (c.boundary chunk.val)
      (a.pid.x*c.BK+key.val) (a.pid.y*c.BV+value.val)
    finalState := fun key value => finalState c data Bool.true a.pid.z
      (a.pid.x*c.BK+key.val) (a.pid.y*c.BV+value.val)
    output := fun time value => outputTile c data a.scale a.pid.z a.pid.y time.val
      (a.pid.x*c.BV+value.val) saved
    dHistory := fun chunk key value => futureState c data a.scale a.pid.z (c.boundary (c.chunks-chunk.val))
      (a.pid.x*c.BK+key.val) (a.pid.y*c.BV+value.val)
    dInitial := fun key value => initialGradient c data a.scale a.pid.z
      (a.pid.x*c.BK+key.val) (a.pid.y*c.BV+value.val)
    dQuery := fun time key => queryTile c data a.scale a.pid.z a.pid.y time.val
      (a.pid.x*c.BK+key.val) saved
    dKey := fun time key => keyTile c data a.scale a.pid.z a.pid.y time.val
      (a.pid.x*c.BK+key.val) future
    dValuePartial := fun block time value => valuePartial c data a.scale a.pid.x a.pid.z a.pid.y time.val
      (block.val*c.BV+value.val) future }

def result (a : KernelPacket.Call) (xs : LogicalTensors.Inputs a.shape) : LogicalTensors.Outputs a.shape :=
  mathOutputs a (LogicalTensors.logicalInputs a xs) (LogicalTensors.history a.shape xs Bool.false) (LogicalTensors.history a.shape xs Bool.true)
end Mathematics


/-- Every history tile and optional final state implements the independent
forward recurrence, with actual-source access safety and the exterior frame.
The original arbitrary program-coordinate domain is retained. -/
specification forward_correct (c : Shape) (seed final : Bool) (pid : PID) :
    Spec.Real (LogicalTensors.io ⟨c,.forwardState,0,seed,final,pid⟩ ⊨ Mathematics.result ⟨c,.forwardState,0,seed,final,pid⟩) :=
  LogicalTensors.correct _ (by simp)

/-- The standalone output kernel takes the saved history as an ordinary input
and implements the independent chunk output contraction. -/
specification output_correct (c : Shape) (scale : ℝ) (pid : PID) :
    Spec.Real (LogicalTensors.io ⟨c,.forwardOutput,scale,Bool.false,Bool.false,pid⟩ ⊨ Mathematics.result ⟨c,.forwardOutput,scale,Bool.false,Bool.false,pid⟩) :=
  LogicalTensors.correct _ (by simp)

/-- Reverse history and optional initial-state gradient. The original state
program geometry remains visible; optional inputs retain their source guards. -/
specification backward_correct (c : Shape) (scale : ℝ) (terminal initial : Bool) (pid : PID)
    (_hp : VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry.StatePID (KernelReal.parameters c) pid) :
    Spec.Real (LogicalTensors.io ⟨c,.backwardState,scale,terminal,initial,pid⟩ ⊨ Mathematics.result ⟨c,.backwardState,scale,terminal,initial,pid⟩) :=
  LogicalTensors.correct _ (by simp)

/-- All dQ, dK and per-key-block partial dV outputs implement the independent
chunk adjoint formulas. The launch installs the original B*H head count;
the original head/chunk conditions are retained without a new key-block bound. -/
specification gradient_correct (c : Shape) (scale : ℝ) (pid : PID)
    (_hh : pid.z<c.B*c.H) (hc : pid.y<c.chunks) :
    Spec.Real (LogicalTensors.io ⟨c,.backwardGradient,scale,Bool.false,Bool.false,pid⟩ ⊨ Mathematics.result ⟨c,.backwardGradient,scale,Bool.false,Bool.false,pid⟩) :=
  LogicalTensors.correct _ (fun _=>hc)

#stmtSurfaceSubset forward_correct ⊆
  [Shape,PID,KernelPacket.Call.mk,KernelPacket.Call.shape,VeriTile.Bench.Repairs.ChunkLinearAttn.Stage.forwardState,
   LogicalTensors.io,Mathematics.result,LogicalProgramIO.Implements,LogicalTensors.Inputs,LogicalTensors.Outputs,Spec.Real]
#stmtSurfaceSubset output_correct ⊆
  [Shape,PID,KernelPacket.Call.mk,KernelPacket.Call.shape,VeriTile.Bench.Repairs.ChunkLinearAttn.Stage.forwardOutput,
   LogicalTensors.io,Mathematics.result,LogicalProgramIO.Implements,LogicalTensors.Inputs,LogicalTensors.Outputs,Spec.Real]
#stmtSurfaceSubset backward_correct ⊆
  [Shape,PID,KernelPacket.Call.mk,KernelPacket.Call.shape,VeriTile.Bench.Repairs.ChunkLinearAttn.Stage.backwardState,
   VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry.StatePID,KernelReal.parameters,
   LogicalTensors.io,Mathematics.result,LogicalProgramIO.Implements,LogicalTensors.Inputs,LogicalTensors.Outputs,Spec.Real]
#stmtSurfaceSubset gradient_correct ⊆
  [Shape,Shape.B,Shape.H,Shape.chunks,PID,VeriTile.Bench.Repairs.ChunkLinearAttn.PID.z,
   VeriTile.Bench.Repairs.ChunkLinearAttn.PID.y,KernelPacket.Call.mk,KernelPacket.Call.shape,
   VeriTile.Bench.Repairs.ChunkLinearAttn.Stage.backwardGradient,
   LogicalTensors.io,Mathematics.result,LogicalProgramIO.Implements,LogicalTensors.Inputs,LogicalTensors.Outputs,Spec.Real]
#specNonCircular Mathematics.result avoiding [FP.Structural.exec,run,KernelPacket.reference,KernelPacket.source]
#axiomsClean forward_correct
#axiomsClean output_correct
#axiomsClean backward_correct
#axiomsClean gradient_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.Correct
