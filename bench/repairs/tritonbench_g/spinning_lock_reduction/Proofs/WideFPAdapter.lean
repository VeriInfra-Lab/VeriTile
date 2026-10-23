import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.WideViews
import bench.tritonbench_g.spinning_lock_reduction.Proofs.FPAdapter

noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.Wide.FPAdapter
open VeriTile Triton Bench.TritonBenchG.SpinningLock

inductive Implementation where | source | orderedReference
inductive Code where | fixedWidthSource | exactViewEncoding | orderedReference

structure Program where
  host : HostArgs
  buffers : Buffers host
  flags : ViewFlags
  implementation : Implementation

abbrev Observation := List Event × Memory (FP.Value .fp32) ×
  (Nat → Nat → FP.Value .fp32) × Unit

def sourceObservation (h : HostArgs) (flags : ViewFlags) (schedule : List Nat)
    (raw : Memory (FP.Value .fp32)) : Observation :=
  let result := physicalLaunch fp32 fpEncoding flags h schedule raw
  (result.events, result.final.values.mem, result.final.values.acc, result.returned)

def referenceObservation (h : HostArgs) (flags : ViewFlags) (schedule : List Nat)
    (raw : Memory (FP.Value .fp32)) : Observation :=
  let logical := interpretMemory fpEncoding flags raw
  let trace := (run fp32 h schedule (initial fp32 logical)).1
  let reference := replay fp32 trace (Reference.initial logical)
  (trace, interpretMemory fpEncoding flags reference.mem,
    fun p l => (reference.reads p l).foldl fp32.add fp32.zero, ())

def eval (program : Program) (schedule : List Nat) (raw : Memory (FP.Value .fp32)) : Observation :=
  match program.implementation with
  | .source => sourceObservation program.host program.flags schedule raw
  | .orderedReference => referenceObservation program.host program.flags schedule raw

def Equivalent (lhs rhs : Program) : Prop := ∀ schedule raw, eval lhs schedule raw = eval rhs schedule raw

instance : Spec.ProgramSyntax Program where
  Statement := Code
  Signature := HostArgs × ViewFlags
  signature p := (p.host, p.flags)
  body p := match p.implementation with
    | .source => [.fixedWidthSource, .exactViewEncoding]
    | .orderedReference => [.orderedReference]
  structural := some Equivalent

def candidate (h : HostArgs) (buf : Buffers h) (flags : ViewFlags) : Program := ⟨h, buf, flags, .source⟩
def reference (h : HostArgs) (buf : Buffers h) (flags : ViewFlags) : Program := ⟨h, buf, flags, .orderedReference⟩

theorem source_reference (h : HostArgs) (buf : Buffers h) (flags : ViewFlags)
    (schedule : List Nat) (raw : Memory (FP.Value .fp32)) :
    sourceObservation h flags schedule raw = referenceObservation h flags schedule raw := by
  obtain ⟨hm, ha⟩ := full_schedule_value_memory fp32 h schedule (interpretMemory fpEncoding flags raw)
  unfold sourceObservation physicalLaunch
  rw [launch_eq_run fp32 h buf]
  change ((run fp32 h schedule (initial fp32 (interpretMemory fpEncoding flags raw))).1,
    interpretMemory fpEncoding flags (run fp32 h schedule (initial fp32 (interpretMemory fpEncoding flags raw))).2.values.mem,
    (run fp32 h schedule (initial fp32 (interpretMemory fpEncoding flags raw))).2.values.acc, ()) = _
  unfold referenceObservation
  rw [hm]
  have hav := funext fun p => funext fun l => ha p l
  rw [hav]

theorem full_fp32_equivalence (h : HostArgs) (buf : Buffers h) (flags : ViewFlags) :
    Spec.FloatingPoint [] (candidate h buf flags) (reference h buf flags) :=
  Spec.FloatingPoint.ofStructural rfl rfl (source_reference h buf flags)

#print_fp_assumptions full_fp32_equivalence
#axiomsClean full_fp32_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Wide.FPAdapter
