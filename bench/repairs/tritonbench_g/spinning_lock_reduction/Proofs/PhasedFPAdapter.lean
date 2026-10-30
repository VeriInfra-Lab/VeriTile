import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.PhasedHost
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.WideFPAdapter

noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.Phased.FPAdapter
open VeriTile Triton Bench.TritonBenchG.SpinningLock Wide

inductive Implementation where | source | orderedReference
inductive Code where | sameStreamStages | fixedWidthSource | exactViewEncoding | orderedReference
structure Program where
  host : HostArgs
  buffers : Buffers host
  flags : ViewFlags
  implementation : Implementation

abbrev Observation := Wide.FPAdapter.Observation

def sourceObservation (h : HostArgs) (flags : ViewFlags) (plan : Plan)
    (raw : Memory (FP.Value .fp32)) : Observation :=
  let result := Phased.observe fp32 fpEncoding flags h plan raw
  (result.events, result.final.values.mem, result.final.values.acc, result.returned)

def referenceObservation (h : HostArgs) (flags : ViewFlags) (plan : Plan)
    (raw : Memory (FP.Value .fp32)) : Observation :=
  Wide.FPAdapter.referenceObservation h flags (flatten h plan) raw

def eval (program : Program) (plan : Plan) (raw : Memory (FP.Value .fp32)) : Observation :=
  match program.implementation with
  | .source => sourceObservation program.host program.flags plan raw
  | .orderedReference => referenceObservation program.host program.flags plan raw

def Equivalent (lhs rhs : Program) : Prop := ∀ plan raw, eval lhs plan raw = eval rhs plan raw

instance : Spec.ProgramSyntax Program where
  Statement := Code
  Signature := HostArgs × ViewFlags
  signature p := (p.host,p.flags)
  body p := match p.implementation with
    | .source => [.sameStreamStages,.fixedWidthSource,.exactViewEncoding]
    | .orderedReference => [.orderedReference]
  structural := some Equivalent

def candidate (h : HostArgs) (buf : Buffers h) (flags : ViewFlags) : Program := ⟨h,buf,flags,.source⟩
def reference (h : HostArgs) (buf : Buffers h) (flags : ViewFlags) : Program := ⟨h,buf,flags,.orderedReference⟩

theorem source_reference (h : HostArgs) (buf : Buffers h) (flags : ViewFlags)
    (plan : Plan) (raw : Memory (FP.Value .fp32)) :
    sourceObservation h flags plan raw = referenceObservation h flags plan raw :=
  Wide.FPAdapter.source_reference h buf flags (flatten h plan) raw

theorem full_fp32_equivalence (h : HostArgs) (buf : Buffers h) (flags : ViewFlags) :
    Spec.FloatingPoint [] (candidate h buf flags) (reference h buf flags) :=
  Spec.FloatingPoint.ofStructural rfl rfl (source_reference h buf flags)

#print_fp_assumptions full_fp32_equivalence
#axiomsClean full_fp32_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Phased.FPAdapter
