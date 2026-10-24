import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.IOResults
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelCompiler

namespace VeriTile.Bench.Repairs.LayerNormOps.Interface
open VeriTile.Triton FP.Structural KernelNative.Shared
noncomputable section

/-- Existing type-directed FP32 lowering and output-specific source ordering.
This pure observer receives one symbolic payload, never an execution state. -/
def nativeValue (I : Instructions) (S : Schedule) (value : Term) : Word :=
  eval I (KernelNative.runtimeSchedule S)
    (normalize (KernelNative.runtimeSchedule S) (KernelNative.lower value))

def forwardObservation (I : Instructions) (S : OutputSchedule) (p : ForwardParameters) (pid : Nat) :
    (o : Fin (forwardOutputs p).length) → TileIndex (forwardOutputs p)[o].shape → Term → Word :=
  fun o j => nativeValue I (S (forwardOutputs p)[o].buf (forwardWrite p pid o j))
def backwardObservation (I : Instructions) (S : OutputSchedule) (p : BackwardParameters) (pid : Nat) :
    (o : Fin (backwardOutputs p).length) → TileIndex (backwardOutputs p)[o].shape → Term → Word :=
  fun o j => nativeValue I (S (backwardOutputs p)[o].buf (backwardWrite p pid o j))

/-- The target evaluates the independent input-array graph with the actual
converted host epsilon and runtime signed-i32 count. Conversion hypotheses are
the pre-existing scalar-only laws; no whole-output equality is assumed. -/
def forwardNative (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (p : ForwardParameters) (pid : Nat) (epsilon : Word) (v : ForwardValues Term p) :
    ChannelValues Word (forwardOutputs p) :=
  fun o j => mapValue
    (forwardObservation (KernelNative.withCount (withEpsilon I p.epsilon epsilon) p.columns) S p pid o j)
    (forwardOutputs p)[o].dtype (forwardGraph (algebra R) p v o j)
def backwardNative (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (p : BackwardParameters) (pid : Nat) (v : BackwardValues Term p) :
    ChannelValues Word (backwardOutputs p) :=
  fun o j => mapValue
    (backwardObservation (KernelNative.withCount I p.columns) S p pid o j)
    (backwardOutputs p)[o].dtype (backwardGraph (algebra R) p pid v o j)

theorem forward_native (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (p : ForwardParameters) (pid : Nat) (admissible : KernelExecution.ForwardAdmissible p)
    (count : KernelNative.CountConversion I p.columns)
    (epsilon : Word) (conversion : literal32 I p.epsilon=epsilon) :
    (forwardIO (algebra R) p pid).Observes (forwardObservation I S p pid)
      (forwardNative R I S p pid epsilon) := by
  have law : KernelNative.withCount (withEpsilon I p.epsilon epsilon) p.columns=I := by
    rw [withEpsilon_eq I p.epsilon epsilon conversion,KernelNative.withCount_eq I p.columns count]
  unfold forwardNative
  rw [law]
  exact ProgramIO.Observes.of_implements (forward_implements (algebra R) p pid admissible)
    (forwardObservation I S p pid)

theorem backward_native (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (p : BackwardParameters) (pid : Nat) (admissible : KernelReadback.Admissible p)
    (count : KernelNative.CountConversion I p.columns) :
    (backwardIO (algebra R) p pid).Observes (backwardObservation I S p pid)
      (backwardNative R I S p pid) := by
  unfold backwardNative
  rw [KernelNative.withCount_eq I p.columns count]
  exact ProgramIO.Observes.of_implements (backward_implements (algebra R) p pid admissible)
    (backwardObservation I S p pid)

#specNonCircular forwardNative avoiding [forwardKernel,FP.Structural.exec,Forward.reference]
#specNonCircular backwardNative avoiding [backwardKernel,FP.Structural.exec,Backward.reference,Backward.update]
#axiomsClean forward_native
#axiomsClean backward_native
end
end VeriTile.Bench.Repairs.LayerNormOps.Interface
