import bench.tritonbench_g.chunk_linear_attn.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Correct
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.LogicalIO
open scoped BigOperators
noncomputable section

/- Independent scalar mathematics and the complete logical output channels.
These definitions are definitionally connected to the canonical target used below. -/
namespace Mathematics
open Real (Inputs)
def state {T K V : Nat} (x : Inputs T K V) (t : Fin T) (e : Fin K) (j : Fin V) : ℝ :=
  x.initial e j + ∑ u : Fin T, if u ≤ t then x.k u e*x.v u j else 0
def output {T K V : Nat} (scale : ℝ) (x : Inputs T K V) (t : Fin T) (j : Fin V) : ℝ :=
  scale * ∑ e : Fin K, x.q t e*state x t e j
def finalState {T K V : Nat} (x : Inputs T K V) (e : Fin K) (j : Fin V) : ℝ :=
  x.initial e j + ∑ u : Fin T, x.k u e*x.v u j
def adjoint {T K V : Nat} (scale : ℝ) (x : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) (u : Fin T) (e : Fin K) (j : Fin V) : ℝ :=
  R e j + scale * ∑ t : Fin T, if u ≤ t then x.q t e*G t j else 0
def gradients {T K V : Nat} (scale : ℝ) (x : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) : Inputs T K V where
  q := fun t e => scale * ∑ j : Fin V, state x t e j*G t j
  k := fun u e => ∑ j : Fin V, adjoint scale x G R u e j*x.v u j
  v := fun u j => ∑ e : Fin K, adjoint scale x G R u e j*x.k u e
  initial := fun e j => R e j + scale * ∑ t : Fin T, x.q t e*G t j

def gradient (c : KernelInterface.Config) (vals : LogicalTensors.Inputs c)
    (head : Fin (c.dims.B*c.dims.H)) : Real.Inputs c.dims.T c.dims.K c.dims.V :=
  gradients c.scale (LogicalTensors.realInputs c vals head) (LogicalTensors.incoming c vals head) (LogicalTensors.terminal c vals head)

def result (c : KernelInterface.Config) (vals : LogicalTensors.Inputs c) : LogicalTensors.Outputs c where
  output := fun head time value => output c.scale (LogicalTensors.realInputs c vals head) time value
  dQuery := fun head time key => (gradient c vals head).q time key
  dKey := fun head time key => (gradient c vals head).k time key
  dValue := fun head time value => (gradient c vals head).v time value
  finalState := fun head key value => finalState (LogicalTensors.realInputs c vals head) key value
  dInitial := fun head key value => (gradient c vals head).initial key value
end Mathematics


/-- Six finite input channels, six result channels, three private buffers,
optional endpoint masks, four complete source grids and the finite FP32 sum. -/
abbrev io (c : KernelInterface.Config) (work : KernelSchedule.Work) := LogicalTensors.io c work

/-- Independent causal attention and its full VJP under every legal grid
permutation. The shared IO judgment includes actual source access safety,
successful execution, all active outputs and the complete exterior frame. -/
specification kernel_correct (c : KernelInterface.Config) (valid : Host.Valid c.dims)
    (work : KernelSchedule.Work) (legal : KernelSchedule.Legal c.call work) :
    Spec.Real ((io c work ⊨ Mathematics.result c) ∧
      ∀ vals head, Real.IsVJP c.scale (LogicalTensors.realInputs c vals head)
        (LogicalTensors.incoming c vals head) (LogicalTensors.terminal c vals head)
        (Mathematics.gradient c vals head)) :=
  ⟨LogicalTensors.correct c work valid legal,
    fun vals head => Mathematical.is_vjp c ((LogicalTensors.inputs c).symm vals) head⟩

#stmtSurfaceSubset kernel_correct ⊆
  [KernelInterface.Config, Host.Valid, KernelSchedule.Work, KernelSchedule.Legal,
    KernelInterface.Config.call, Spec.Real, io, LogicalIO.Implements,
    Mathematics.result, Real.IsVJP, LogicalTensors.realInputs, LogicalTensors.incoming,
    LogicalTensors.terminal, Mathematics.gradient, LogicalTensors.Inputs, LogicalTensors.Outputs,
    ChannelValues, IOLayout.inputs, IOLayout.outputs, ReductionProgramIO.Implements, IO.real,
    KernelInterface.Config.dims, KernelInterface.Config.scale, Host.Dimensions.B,
    Host.Dimensions.H, Host.Dimensions.T, Host.Dimensions.K, Host.Dimensions.V]
#specNonCircular Mathematics.result avoiding [KernelSchedule.run,KernelExecution.run,
  FP.Structural.exec,runGrid,executeFour,gridMemory,stageMemory,modelFour,completeMemory]
#axiomsClean kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Correct
