import bench.tritonbench_g.chunk_retention.Kernels
import bench.tritonbench_g.chunk_retention.Spec
import bench.tritonbench_g.chunk_retention.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelOnly
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
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

def future (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head time key value : Nat) :=
  suffix (headRate c head) (xs.finalIncoming head key value)
    (forcing xs scale head key value) time (c.T-time)

def finalState (c : Shape) (xs : Inputs ℝ) (hasSeed : Bool)
    (head key value : Nat) : ℝ := state c xs hasSeed head c.T key value

def initialGradient (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head key value : Nat) : ℝ := future c xs scale head 0 key value

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

def result (a : Call) (p : Nat→Nat) (xs : LogicalTensors.Inputs a.shape) : LogicalTensors.Outputs a.shape where
  history := fun chunk key value => state a.shape (LogicalTensors.logicalInputs a xs) Bool.true (p 2)
      (chunk.val*64) (p 0*a.shape.BK+key.val) (p 1*a.shape.BV+value.val)
  finalState := fun key value => finalState a.shape (LogicalTensors.logicalInputs a xs) Bool.true (p 2)
      (p 0*a.shape.BK+key.val) (p 1*a.shape.BV+value.val)
  output := fun time value => outputTile a.shape (LogicalTensors.logicalInputs a xs) a.scale (p 2) (p 1)
      time.val (p 0*a.shape.BV+value.val) (LogicalTensors.history a.shape xs Bool.false (p 2) (p 1))
  dHistory := fun chunk key value => future a.shape (LogicalTensors.logicalInputs a xs) a.scale (p 2)
      (a.shape.boundary (a.shape.chunks-chunk.val))
      (p 0*a.shape.BK+key.val) (p 1*a.shape.BV+value.val)
  dQuery := fun time key => queryTile a.shape (LogicalTensors.logicalInputs a xs) a.scale (p 2) (p 1)
      time.val (p 0*a.shape.BK+key.val) (LogicalTensors.history a.shape xs Bool.false (p 2) (p 1))
  dKey := fun time key => keyTile a.shape (LogicalTensors.logicalInputs a xs) a.scale (p 2) (p 1)
      time.val (p 0*a.shape.BK+key.val) (LogicalTensors.history a.shape xs Bool.true (p 2) (p 1))
  dValuePartial := fun block time value => valuePartial a.shape (LogicalTensors.logicalInputs a xs) a.scale
      (p 0) (p 2) (p 1) time.val (block.val*a.shape.BV+value.val)
      (LogicalTensors.history a.shape xs Bool.true (p 2) (p 1))
end Mathematics


/-- Each of the four selected FP32 kernels implements its independent Real
retention/VJP formula. IO.lean exposes all inputs, seven possible outputs,
original coordinate guards, source masks, and the exact exterior frame. -/
specification source_correct (a : Call) (p grid : Nat→Nat) (legal : PublicIO.Legal a p grid) :
    Spec.Real (LogicalTensors.io a p grid ⊨ Mathematics.result a p) := by
  exact LogicalTensors.correct a p grid legal

#stmtSurfaceSubset source_correct ⊆
  [Call,KernelPacket.Call,KernelPacket.Call.shape,PublicIO.Legal,Spec.Real,LogicalTensors.io,LogicalProgramIO.Implements,LogicalTensors.Inputs,LogicalTensors.Outputs,
   Mathematics.result]
#axiomsClean source_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelOnly
