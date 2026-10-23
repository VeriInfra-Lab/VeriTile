import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperCompiledSelection

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper NativeMachine
noncomputable section

structure CompileKey where
  sourceSHA256 : String
  arguments : LaunchArguments
  compilerBuild : String
  target : String
  numWarps : Nat
  numStages : Nat
  enableFPFusion : Bool

/-- Raw compiler result. Its identity names the actual emitted artifact;
trees, local dependency expressions and store descriptors are its projections. -/
structure Emission (block : Nat) where
  identity : String
  schedule : Schedule block
  code : Compiler.Root → Compiler.Code
  stores : Compiler.Destination → Option Compiler.StoreOperand

/-- Exact selected compilation context. This restricts the compiler contract,
not the Python call domain; other output-storage branches are handled by their
own explicit conversion profile. No numerical operand restriction occurs. -/
structure CompilerKeyFacts (compilerBuild target : String) (key : CompileKey) : Prop where
  source : key.sourceSHA256=Compiler.sourceSHA256
  build : key.compilerBuild=compilerBuild
  target : key.target=target
  warps : key.numWarps ∈ [1,2,4,8,16,32]
  stages : key.numStages=3
  fusion : key.enableFPFusion=Bool.true
  storage : SelectedStorage key.arguments.request key.arguments.prepared

/-- Ordinary compiler implementation boundary. The only conformance fields
describe individual primitive operands, source origins and store descriptors
of the actual successful emitted result. There is no numerical, output,
memory-preservation or successful-kernel-execution equation. -/
structure CompilerAPI (Host : Type) where
  compilerBuild : String
  target : String
  compile : (key : CompileKey) → Host → Except Failure (Emission key.arguments.prepared.block)
  operands : ∀ key host emitted, compile key host=.ok emitted →
    CompilerKeyFacts compilerBuild target key →
    ∀ root, root.active (key.arguments.prepared.flags key.arguments.request) key.arguments.prepared.validated.N →
      Compiler.Binds (Compiler.recipe emitted.schedule (key.arguments.prepared.flags key.arguments.request)
        key.arguments.prepared.validated.N root) (emitted.code root)
  stores : ∀ key host emitted, compile key host=.ok emitted →
    CompilerKeyFacts compilerBuild target key →
    ∀ destination, emitted.stores destination=
      Compiler.sourceStore (key.arguments.prepared.flags key.arguments.request)
        key.arguments.prepared.validated.N emitted.code destination

def CompilerAPI.key (compiler : CompilerAPI Host) (args : LaunchArguments) (warps : Nat) : CompileKey :=
  ⟨Compiler.sourceSHA256,args,compiler.compilerBuild,compiler.target,warps,3,Bool.true⟩

/-- A runtime-selected profile retains the actual compiler call and result.
The num_warps choice enters the compile key, and its returned schedule is used
without replacing any owner tree by a different artifact's tree. -/
structure Produced (compiler : CompilerAPI Host) (args : LaunchArguments) (host : Host) where
  warps : Nat
  supported : warps ∈ [1,2,4,8,16,32]
  storage : SelectedStorage args.request args.prepared
  emission : Emission args.prepared.block
  compiled : compiler.compile (compiler.key args warps) host=.ok emission

def Produced.artifact {compiler : CompilerAPI Host} {args : LaunchArguments} {host : Host}
    (produced : Produced compiler args host) :
    Compiler.Artifact (args.prepared.flags args.request) args.prepared.validated.N args.prepared.block where
  warps := produced.warps
  supportedWarps := produced.supported
  schedule := produced.emission.schedule
  code := produced.emission.code
  stores := produced.emission.stores
  operands := compiler.operands (compiler.key args produced.warps) host produced.emission produced.compiled
    ⟨rfl,rfl,rfl,produced.supported,rfl,rfl,produced.storage⟩
  storeOperands := compiler.stores (compiler.key args produced.warps) host produced.emission produced.compiled
    ⟨rfl,rfl,rfl,produced.supported,rfl,rfl,produced.storage⟩

theorem Produced.provenance {compiler : CompilerAPI Host} {args : LaunchArguments} {host : Host}
    (produced : Produced compiler args host) :
    compiler.compile (compiler.key args produced.artifact.warps) host=.ok produced.emission ∧
    produced.artifact.schedule.reductions=produced.emission.schedule.reductions ∧
    produced.artifact.code=produced.emission.code ∧ produced.artifact.stores=produced.emission.stores :=
  ⟨produced.compiled,rfl,rfl,rfl⟩

inductive ProducedEvent (compiler : CompilerAPI Host) (args : LaunchArguments) (host : Host) where
  | launch (produced : Produced compiler args host)
  | setupFailure (code : Nat)

def ProducedEvent.compiled {compiler : CompilerAPI Host} {args : LaunchArguments} {host : Host} :
    ProducedEvent compiler args host →
      CompiledEvent (args.prepared.flags args.request) args.prepared.validated.N args.prepared.block
  | .launch produced => .launch produced.artifact
  | .setupFailure code => .setupFailure code

structure ProducedDispatch (compiler : CompilerAPI Host) (args : LaunchArguments) (host : Host) where
  benchmarks : List (ProducedEvent compiler args host)
  selected : Produced compiler args host

def ProducedDispatch.compiled {compiler : CompilerAPI Host} {args : LaunchArguments} {host : Host}
    (produced : ProducedDispatch compiler args host) :
    CompiledDispatch (args.prepared.flags args.request) args.prepared.validated.N args.prepared.block :=
  ⟨produced.benchmarks.map ProducedEvent.compiled,produced.selected.artifact⟩

structure ProducedSelection (Host : Type) where
  compiler : CompilerAPI Host
  select : (args : LaunchArguments) → (host : Host) → Except Failure (ProducedDispatch compiler args host)

def ProducedSelection.toCompiled (selection : ProducedSelection Host) : CompiledSelection Host where
  compile source args host :=
    if source=Compiler.sourceSHA256 then do
      let produced ← selection.select args host
      pure produced.compiled
    else .error (.api .admission 0)

theorem ProducedSelection.selected (selection : ProducedSelection Host) (args : LaunchArguments)
    (host : Host) (compiled : CompiledDispatch (args.prepared.flags args.request) args.prepared.validated.N args.prepared.block)
    (h : selection.toCompiled.compile Compiler.sourceSHA256 args host=.ok compiled) :
    ∃ produced, selection.select args host=.ok produced ∧ compiled=produced.compiled := by
  cases hs : selection.select args host with
  | error e => simp [toCompiled,hs] at h
  | ok produced =>
    have he : produced.compiled=compiled := by simpa [toCompiled,hs] using h
    exact ⟨produced,rfl,he.symm⟩

#axiomsClean Produced.provenance
#axiomsClean ProducedSelection.selected
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
