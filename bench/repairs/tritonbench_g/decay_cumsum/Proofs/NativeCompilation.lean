import bench.repairs.tritonbench_g.decay_cumsum.Proofs.NativeLowering
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.StructuralPublic

namespace VeriTile.Bench.Repairs.DecayCumsum.Native
open VeriTile.Triton FP.Structural PackedStorage
noncomputable section
set_option maxHeartbeats 5000000

def candidateHash : String := "8280a38504ba5e0d5fb398180cc30cb90230739a6ca545e2c455e1d6cc309974"

structure Configuration where
  kind : Kind
  B : Nat
  H : Nat
  T : Nat
  headStride : Nat
  width : Nat
  BT : Nat
  BK : Nat
  deriving DecidableEq
def configuration (h : Call) (kind : Kind) (phase : Phase) : Configuration :=
  ⟨kind,h.B,h.H,if phase.span=h.BT then h.T else phase.span,
    h.headStride,h.width,phase.span,h.BK⟩

/-- Bind the actual main/tail runtime T argument. The numerical source does
not read T, but its launch ABI still records the real supplied argument. -/
def sourceTemplate (h : Call) (kind : Kind) (phase : Phase) : ComputeKernel :=
  let runtimeT := (configuration h kind phase).T
  match kind with
  | .forward => Original.forward .fp32 Bool.false "g" "go"
      h.headStride h.width 1 h.B h.H runtimeT phase.span h.BK h.width scaleSlot
  | .prepare => Original.prepare .fp32 .fp32 Bool.false "q" "k" "g" "qg" "kg"
      h.headStride h.width 1 h.B h.H runtimeT phase.span h.BK h.width scaleSlot
  | .backward => Original.backward .fp32 .fp32 Bool.false
      "dqi" "dqo" "dki" "dko" "q" "k" "g" "dg"
      h.headStride h.width 1 h.B h.H runtimeT phase.span h.BK h.width scaleSlot

theorem source_template_eq (h : Call) (kind : Kind) (phase : Phase) :
    sourceTemplate h kind phase=Kernels.source kind h phase scaleSlot := by
  cases kind <;> rfl

/-- The source template's reserved scale occurrence is a raw fp32 parameter,
not a floating constant or a projection through ℝ. Literal zero and inv_ln2
are distinct syntax roles. Cast/load/store are FP32 bit-preserving boundaries. -/
inductive ScalarABI where
  | fp32RuntimeScaleSlotWithBitPreservingMemory
  | other (description : String)
  deriving DecidableEq

structure Artifact where
  sourceHash : String
  template : ComputeKernel
  configuration : Configuration
  backend : String
  architecture : Nat
  compiler : String
  enableFPFusion : Bool
  numWarps : Nat
  numStages : Nat
  numCTAs : Nat
  scalarABI : ScalarABI
structure Resolution where
  artifact : Artifact
  instructions : Instructions
  /-- A finite local instruction-expression graph for each SOURCE SITE.
  It cannot inspect a runtime tensor or return an output array. -/
  graph : Graph
structure Compiler where
  resolve : Call → Kind → Phase → Resolution

/-- A local compilation/ABI contract for this exact source and configuration.
Only syntax and metadata are compared; no execution or output equality is a
field. The finite saved profiles support this boundary, not its universal
truth for uncompiled future configurations. -/
structure Bound (compiler : Compiler) (p : Instructions) (h : Call) (kind : Kind) (phase : Phase) : Prop where
  source : (compiler.resolve h kind phase).artifact.sourceHash=candidateHash
  template : (compiler.resolve h kind phase).artifact.template=sourceTemplate h kind phase
  config : (compiler.resolve h kind phase).artifact.configuration=configuration h kind phase
  backend : (compiler.resolve h kind phase).artifact.backend="cuda"
  architecture : (compiler.resolve h kind phase).artifact.architecture=90
  version : (compiler.resolve h kind phase).artifact.compiler="3.7.1"
  fusion : (compiler.resolve h kind phase).artifact.enableFPFusion=Bool.true
  warps : (compiler.resolve h kind phase).artifact.numWarps=4
  stages : (compiler.resolve h kind phase).artifact.numStages=3
  ctas : (compiler.resolve h kind phase).artifact.numCTAs=1
  scalarABI : (compiler.resolve h kind phase).artifact.scalarABI=.fp32RuntimeScaleSlotWithBitPreservingMemory
  instructions : (compiler.resolve h kind phase).instructions=p
  sites : ∀ site, (compiler.resolve h kind phase).graph site=selected site
  arange : NativeABI.CompiledArange h

/-- Compilation facts are required only after successful preparation AND
setup. Empty calls and every preparation/setup failure retain their full domain. -/
structure LocalCompilation {β : Type} (compiler : Compiler) (p : Instructions)
    (h : Call) (kind : Kind) (api : WholeHost.API h kind β (Cell Node)) : Prop where
  setup : ∀ prepared, api.prepare=.ok prepared → api.setup prepared=.ok () →
    ∀ w∈Grid.schedule h, Bound compiler p h kind w.1

def compiledExecute (compiler : Compiler) (h : Call) (kind : Kind)
    (storage : Placement h kind) (scale : Word) (w : Grid.Work h) :
    FPStorage.Heap Node → Option (FPStorage.Heap Node) :=
  let result := compiler.resolve h kind w.1
  FPStorage.execute (algebra result.instructions result.graph kind scale) h kind storage scaleSlot w

theorem compiled_execute_eq (compiler : Compiler) (p : Instructions) (h : Call) (kind : Kind)
    (storage : Placement h kind) (scale : Word) (w : Grid.Work h)
    (binding : Bound compiler p h kind w.1) :
    compiledExecute compiler h kind storage scale w=
      FPStorage.execute (algebra p selected kind scale) h kind storage scaleSlot w := by
  have graph : (compiler.resolve h kind w.1).graph=selected := funext binding.sites
  simp only [compiledExecute,binding.instructions,graph]

theorem run_congr {W β : Type} (a b : W → Run.Heap β → Option (Run.Heap β)) (jobs : List W)
    (same : ∀ w∈jobs, a w=b w) (heap : Run.Heap β) : Run.run a jobs heap=Run.run b jobs heap := by
  induction jobs generalizing heap with
  | nil => rfl
  | cons w ws ih =>
    have head := same w (by simp)
    have tail := ih (fun v hv => same v (List.mem_cons_of_mem w hv))
    simp only [Run.run,head]
    congr 1
    funext next
    exact tail next

def compiledRun {β : Type} (compiler : Compiler) (h : Call) (kind : Kind) (scale : Word)
    (api : WholeHost.API h kind β (Cell Node)) (targets : RegionName → Host.Target)
    (encode : Bool → Cell Node → Fin 4 → β) (before : Copyback.Heap β) :=
  WholeHost.run h kind api targets encode
    (fun prepared => Run.run (compiledExecute compiler h kind prepared.storage scale) (Grid.schedule h))
    (fun prepared => LogicalStorage.crop h kind prepared.storage) before

/-- Compiler selection is consumed by the actual run, rather than attached as
an unused evidence premise. Setup failure does not demand a compilation. -/
theorem compiled_run_eq {β : Type} (compiler : Compiler) (p : Instructions) (h : Call) (kind : Kind) (scale : Word)
    (api : WholeHost.API h kind β (Cell Node)) (targets : RegionName → Host.Target)
    (encode : Bool → Cell Node → Fin 4 → β) (before : Copyback.Heap β)
    (localCompilation : LocalCompilation compiler p h kind api) :
    compiledRun compiler h kind scale api targets encode before=
      WholeHost.run h kind api targets encode
        (fun prepared => Run.run (FPStorage.execute (algebra p selected kind scale) h kind prepared.storage scaleSlot) (Grid.schedule h))
        (fun prepared => LogicalStorage.crop h kind prepared.storage) before := by
  unfold compiledRun WholeHost.run
  by_cases empty : Logical.cells h=0
  · simp [empty]
  · simp only [if_neg empty]
    cases preparedEq : api.prepare with
    | error message => rfl
    | ok prepared =>
      simp only [Host.run,if_neg empty,WholeHost.preparedAPI]
      cases setupEq : api.setup prepared with
      | error message => rfl
      | ok done =>
        cases done
        have grids := run_congr (compiledExecute compiler h kind prepared.storage scale)
          (FPStorage.execute (algebra p selected kind scale) h kind prepared.storage scaleSlot)
          (Grid.schedule h)
          (fun w hw => compiled_execute_eq compiler p h kind prepared.storage scale w
            (localCompilation.setup prepared preparedEq setupEq w hw)) prepared.heap
        rw [grids]

#axiomsClean compiled_execute_eq
#axiomsClean compiled_run_eq
#axiomsClean source_template_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DecayCumsum.Native
