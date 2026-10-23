import bench.tritonbench_g.spinning_lock_reduction.Proofs.FP
import VeriTile.Spec

/-! Reader adapter for the existing full-source FP32 proof. The reference has
no program counter or accumulator: it folds its own ordered read lists. The
public observation includes the whole P/C/lock memories, every accumulator,
the control-generated event list and the wrapper's None return. -/
namespace VeriTile.Bench.TritonBenchG.SpinningLock.FPAdapter
open VeriTile Triton Frontend Kernels

inductive Implementation where
  | source | orderedReference

inductive Code where
  | source (statement : Statement)
  | orderedReference

structure Program where
  host : HostArgs
  implementation : Implementation

abbrev Observation := List Event × Memory (FP.Value .fp32) ×
  (Nat → Nat → FP.Value .fp32) × Option Unit

def referenceObservation (h : HostArgs) (schedule : List Nat)
    (mem : Memory (FP.Value .fp32)) : Observation :=
  let events := controlTrace originalPlan h.config schedule mem.locks
  let reference := replay fp32 events (Reference.initial mem)
  (events, reference.mem, fun p l => (reference.reads p l).foldl fp32.add fp32.zero, none)

def eval (program : Program) (schedule : List Nat) (mem : Memory (FP.Value .fp32)) :
    Option Observation :=
  match program.implementation with
  | .source => (launch fp32 program.host schedule mem).map fun result =>
      (result.events, result.final.mem, result.final.acc, result.returned)
  | .orderedReference => some (referenceObservation program.host schedule mem)

/-- A structural step requires successful evaluations with identical full
observations, for every finite schedule and arbitrary initial memory. -/
def Equivalent (lhs rhs : Program) : Prop :=
  ∀ schedule mem, ∃ observation,
    eval lhs schedule mem = some observation ∧ eval rhs schedule mem = some observation

instance : Spec.ProgramSyntax Program where
  Statement := Code
  Signature := HostArgs
  signature := Program.host
  body := fun p => match p.implementation with
    | .source => originalTypedSource.map Code.source
    | .orderedReference => [.orderedReference]
  structural := some Equivalent

def original (h : HostArgs) : Program := ⟨h, .source⟩
def reference (h : HostArgs) : Program := ⟨h, .orderedReference⟩

 theorem original_reference (h : HostArgs) : Equivalent (original h) (reference h) := by
  intro schedule mem
  obtain ⟨final, he, hm, hv, _⟩ := SourceCorrect.compiled_source_fp32 h.config schedule mem
  refine ⟨referenceObservation h schedule mem, ?_, rfl⟩
  have ha : final.acc = fun p l =>
      ((replay fp32 (controlTrace originalPlan h.config schedule mem.locks)
        (Reference.initial mem)).reads p l).foldl fp32.add fp32.zero :=
    funext fun p => funext fun l => hv p l
  simp only [eval, original, launch, originalSource, he, Option.map_some, wrapperReturn]
  simp only [referenceObservation, hm, ha]

 theorem full_fp32_equivalence (h : HostArgs) :
    Spec.FloatingPoint [] (original h) (reference h) :=
  Spec.FloatingPoint.ofStructural rfl rfl (original_reference h)

#print_fp_assumptions full_fp32_equivalence
#axiomsClean full_fp32_equivalence
end VeriTile.Bench.TritonBenchG.SpinningLock.FPAdapter
