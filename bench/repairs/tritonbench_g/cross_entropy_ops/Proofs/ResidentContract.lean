import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Host

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Resident
open CrossEntropy1.Host (Step)

/-- A language is installed once. In particular `run` is not an argument to
an individual refinement application. Concrete adapters must expose a fixed
AST and its ordinary primitive/control/memory interpreter. -/
structure SourceLanguage (NativeState : Type) where
  Program : Type
  Syntax : Type
  Model : Type
  State : Model → Type
  Result : Program → Type
  ABI : Program → Type
  Metadata : Program → Type
  ast : Program → Syntax
  sourceDigest : Program → String
  native : (model : Model) → State model → NativeState
  run : (model : Model) → (program : Program) →
    State model → Step (State model) (Result program)
  coherent : (model : Model) → (program : Program) → ABI program → State model → Prop
  abiMetadata : (program : Program) → ABI program → Metadata program
  metadata : (model : Model) → (program : Program) → (initial : State model) →
    (result : Result program) → (final : State model) →
    run model program initial=.ok result final → Metadata program

/-- One joint installation key. The fixed AST and digest enter the resident
assertion itself alongside the complete ABI. -/
structure ResidentKey {σ : Type} (language : SourceLanguage σ) (program : language.Program) where
  ast : language.Syntax
  sourceDigest : String
  abi : language.ABI program

def SourceLanguage.key {σ : Type} (language : SourceLanguage σ)
    (program : language.Program) (abi : language.ABI program) : ResidentKey language program :=
  ⟨language.ast program,language.sourceDigest program,abi⟩

/-- Ordinary installed compiler/runtime interface. `models` is the fixed
per-primitive runtime-model relation; it must not contain task postconditions.
The execution records the actual returned descriptor/value, final native
state, and one source-labelled origin trace from this same artifact. -/
structure InstalledCompiler {σ : Type} (language : SourceLanguage σ) where
  Artifact : Type
  Trace : Type
  resident : (program : language.Program) → ResidentKey language program → Artifact → Prop
  models : language.Model → Artifact → Prop
  execute : (program : language.Program) → Artifact → language.ABI program →
    σ → language.Result program → σ → Trace → Prop

/-- A representation belongs to the fixed language adapter. It is never a
task-output predicate passed to `simulate`. The numeric instance reads the
one execution's fixed provenance table; the marker instance preserves opaque
old-cell tokens. -/
structure Representations {σ : Type} (language : SourceLanguage σ)
    (compiler : InstalledCompiler language) where
  initial : (model : language.Model) → (program : language.Program) →
    compiler.Artifact → language.ABI program → compiler.Trace → language.State model → Prop
  final : (model : language.Model) → (program : language.Program) →
    compiler.Artifact → language.ABI program → compiler.Trace → language.State model →
      language.Result program → language.State model → Prop

/-- Generic successful-program backward simulation. It constructs a run of
the fixed source interpreter and preserves the actual returned value as well
as storage representation. There are no algorithm formulas, per-output graph
certificates, or caller exclusions in this interface. -/
structure ResidentRefinement {σ : Type} (language : SourceLanguage σ)
    (compiler : InstalledCompiler language) (representation : Representations language compiler) : Prop where
  simulate : ∀ model program artifact abi initial nativeResult nativeFinal trace,
    compiler.resident program (language.key program abi) artifact →
    compiler.models model artifact →
    language.coherent model program abi initial →
    representation.initial model program artifact abi trace initial →
    compiler.execute program artifact abi (language.native model initial) nativeResult nativeFinal trace →
    ∃ final, ∃ success : language.run model program initial=.ok nativeResult final,
      language.native model final=nativeFinal ∧
      language.metadata model program initial nativeResult final success=language.abiMetadata program abi ∧
      representation.final model program artifact abi trace initial nativeResult final

end VeriTile.Bench.Repairs.CrossEntropyOps.Resident
