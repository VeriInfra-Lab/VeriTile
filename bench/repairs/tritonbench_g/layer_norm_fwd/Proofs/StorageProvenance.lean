import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.StorageCompiler
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.CompilerProvenance

namespace VeriTile.Bench.Repairs.LayerNormFwd.StorageCompiler
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper
noncomputable section
set_option maxHeartbeats 2000000

/-- Actual prepared input and requested output metadata, derived from the host
prefix. Only unspecified floating inputs default to FP32. -/
def RequestedStorage (args : LaunchArguments) : Prop :=
  ∀ role tensor, WrapperNative.roleTensor args.prepared role=some tensor →
    tensor.dtype=WrapperNative.requestedRoleDType args.request role

structure KeyFacts (compilerBuild target : String) (key : WrapperNative.CompileKey) : Prop where
  source : key.sourceSHA256=Compiler.sourceSHA256
  build : key.compilerBuild=compilerBuild
  target : key.target=target
  warps : key.numWarps ∈ [1,2,4,8,16,32]
  stages : key.numStages=3
  fusion : key.enableFPFusion=Bool.true
  storage : RequestedStorage key.arguments

structure Emission (block : Nat) where
  identity : String
  schedule : NativeMachine.Schedule block
  code : Compiler.Root → Compiler.Code
  stores : Compiler.Destination → Option Operand

/-- The compiler boundary reports only actual instruction operand bindings
and typed store descriptors on its successful result. Conversion, memory and
return values are computed by the execution proofs, not supplied here. -/
structure CompilerAPI (Host : Type) where
  compilerBuild : String
  target : String
  compile : (key : WrapperNative.CompileKey) → Host → Except Failure (Emission key.arguments.prepared.block)
  operands : ∀ key host emitted, compile key host=.ok emitted →
    KeyFacts compilerBuild target key →
    ∀ root, root.active (key.arguments.prepared.flags key.arguments.request) key.arguments.prepared.validated.N →
      Compiler.Binds (Compiler.recipe emitted.schedule (key.arguments.prepared.flags key.arguments.request)
        key.arguments.prepared.validated.N root) (emitted.code root)
  stores : ∀ key host emitted, compile key host=.ok emitted →
    KeyFacts compilerBuild target key →
    ∀ destination, emitted.stores destination=
      sourceStore (Profile.ofRequest key.arguments.request) (key.arguments.prepared.flags key.arguments.request)
        key.arguments.prepared.validated.N emitted.code destination

def CompilerAPI.key (compiler : CompilerAPI Host) (args : LaunchArguments) (warps : Nat) : WrapperNative.CompileKey :=
  ⟨Compiler.sourceSHA256,args,compiler.compilerBuild,compiler.target,warps,3,Bool.true⟩

structure Produced (compiler : CompilerAPI Host) (args : LaunchArguments) (host : Host) where
  warps : Nat
  supported : warps ∈ [1,2,4,8,16,32]
  storage : RequestedStorage args
  emission : Emission args.prepared.block
  compiled : compiler.compile (compiler.key args warps) host=.ok emission

def Produced.artifact {compiler : CompilerAPI Host} {args : LaunchArguments} {host : Host}
    (produced : Produced compiler args host) :
    Artifact (Profile.ofRequest args.request) (args.prepared.flags args.request)
      args.prepared.validated.N args.prepared.block where
  graph := {
    warps := produced.warps
    supportedWarps := produced.supported
    schedule := produced.emission.schedule
    code := produced.emission.code
    stores := fun destination => (produced.emission.stores destination).map Operand.erase
    operands := compiler.operands (compiler.key args produced.warps) host produced.emission produced.compiled
      ⟨rfl,rfl,rfl,produced.supported,rfl,rfl,produced.storage⟩
    storeOperands := by
      intro destination
      rw [compiler.stores (compiler.key args produced.warps) host produced.emission produced.compiled
        ⟨rfl,rfl,rfl,produced.supported,rfl,rfl,produced.storage⟩ destination]
      dsimp only [CompilerAPI.key]
      cases h : Compiler.sourceStore (args.prepared.flags args.request) args.prepared.validated.N produced.emission.code destination with
      | none => simp [sourceStore,h]
      | some operand => cases operand <;> simp [sourceStore,h,Operand.ofBase,Operand.erase] }
  stores := produced.emission.stores
  storeOperands := compiler.stores (compiler.key args produced.warps) host produced.emission produced.compiled
    ⟨rfl,rfl,rfl,produced.supported,rfl,rfl,produced.storage⟩

theorem Produced.provenance {compiler : CompilerAPI Host} {args : LaunchArguments} {host : Host}
    (produced : Produced compiler args host) :
    compiler.compile (compiler.key args produced.artifact.graph.warps) host=.ok produced.emission ∧
    produced.artifact.graph.schedule=produced.emission.schedule ∧
    produced.artifact.graph.code=produced.emission.code ∧ produced.artifact.stores=produced.emission.stores :=
  ⟨produced.compiled,rfl,rfl,rfl⟩

inductive ProducedEvent (compiler : CompilerAPI Host) (args : LaunchArguments) (host : Host) where
  | launch (produced : Produced compiler args host)
  | setupFailure (code : Nat)

structure ProducedDispatch (compiler : CompilerAPI Host) (args : LaunchArguments) (host : Host) where
  benchmarks : List (ProducedEvent compiler args host)
  selected : Produced compiler args host

structure ProducedSelection (Host : Type) where
  compiler : CompilerAPI Host
  select : (args : LaunchArguments) → (host : Host) → Except Failure (ProducedDispatch compiler args host)

#axiomsClean Produced.provenance
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.StorageCompiler
