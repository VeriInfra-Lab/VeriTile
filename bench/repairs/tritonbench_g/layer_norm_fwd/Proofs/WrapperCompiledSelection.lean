import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.CompilerProfile
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperConcreteFamily

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper NativeMachine
noncomputable section

inductive CompiledEvent (f : Flags) (N block : Nat) where
  | launch (artifact : Compiler.Artifact f N block)
  | setupFailure (code : Nat)

def CompiledEvent.schedule : CompiledEvent f N block → DispatchEvent block
  | .launch artifact => .launch artifact.schedule
  | .setupFailure code => .setupFailure code

/-- Every benchmark and the selected launch retain their own emitted operand
trees and source-site bindings. No profiling output or tensor is supplied. -/
structure CompiledDispatch (f : Flags) (N block : Nat) where
  benchmarks : List (CompiledEvent f N block)
  selected : Compiler.Artifact f N block

def CompiledDispatch.schedules (compiled : CompiledDispatch f N block) : DispatchSchedules block :=
  ⟨compiled.benchmarks.map CompiledEvent.schedule,compiled.selected.schedule⟩

/-- Ordinary compiler/runtime interface for the pinned actual source and the
actual wrapper arguments. The compiler contract consists only of local
instruction/operand/source-site/store bindings in each returned artifact.
Errors remain ordinary setup outcomes. No successful execution is assumed. -/
structure CompiledSelection (Host : Type) where
  compile : (sourceSHA256 : String) → (args : LaunchArguments) → Host →
    Except Failure (CompiledDispatch (args.prepared.flags args.request)
      args.prepared.validated.N args.prepared.block)

def CompiledSelection.toSchedules (selection : CompiledSelection Host) : ScheduleSelection Host where
  select args host := do
    let compiled ← selection.compile Compiler.sourceSHA256 args host
    pure compiled.schedules

theorem CompiledSelection.selected (selection : CompiledSelection Host) (args : LaunchArguments)
    (host : Host) (schedules : DispatchSchedules args.prepared.block)
    (h : selection.toSchedules.select args host=.ok schedules) :
    ∃ compiled, selection.compile Compiler.sourceSHA256 args host=.ok compiled ∧
      schedules=compiled.schedules := by
  cases hc : selection.compile Compiler.sourceSHA256 args host with
  | error e => simp [toSchedules,hc] at h
  | ok compiled =>
    have he : compiled.schedules=schedules := by simpa [toSchedules,hc] using h
    exact ⟨compiled,rfl,he.symm⟩

theorem CompiledSelection.failure (selection : CompiledSelection Host) (args : LaunchArguments)
    (host : Host) (error : Failure)
    (h : selection.compile Compiler.sourceSHA256 args host=.error error) :
    selection.toSchedules.select args host=.error error := by
  simp [toSchedules,h]

#axiomsClean CompiledSelection.selected
#axiomsClean CompiledSelection.failure
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
