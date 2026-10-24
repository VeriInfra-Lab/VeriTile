import bench.tritonbench_g.spinning_lock_reduction.Kernels
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.PhasedFour
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.PhasedProgress

/-! Historical publication-only adapter, preserved for reuse and frozen evidence.
The final wide/view-aware public adapter follows below.
Separate one-line repair of source SHA256
1542d9675258944fa64fd3688b5ecd1a3116a9c4e277b38294093d133cadf20c.
The complete original typed tree is retained; a block barrier is inserted before
the sole producer exchange. The external AST checker binds this transformation
to `spinning_lock_reduction_repaired.py`. P/C, loads, stores and additions remain
FP32; integer expressions remain the original signed-i32 expressions.

`project` is the whole-tile SC value projection, not an implicit native barrier
assumption. The barrier is visible in `candidateSource`; its ordinary Triton/PTX
ordering and actual offline lowering are reviewed in the repair evidence.
-/
namespace VeriTile.Bench.Repairs.SpinningLock.Kernels
open VeriTile.Bench.TritonBenchG.SpinningLock
open Frontend

inductive SynchronizedStatement where
  | ordinary (statement : Statement)
  | forRange (index : String) (start stop : Nat) (body : List SynchronizedStatement)
  | branch (test : BExpr) (yes no : List SynchronizedStatement)
  | whileLoop (test : Condition) (body : List SynchronizedStatement)
  | sequence (body : List SynchronizedStatement)
  | blockBarrier

/-- The only inserted instruction is a CTA barrier before each producer release;
there is exactly one such source site, within the unchanged nine-round loop. -/
def synchronize : Statement → SynchronizedStatement
  | .forRange i a b body => .forRange i a b (body.map synchronize)
  | .branch c yes no => .branch c (yes.map synchronize) (no.map synchronize)
  | .whileLoop c body => .whileLoop c (body.map synchronize)
  | s@(.xchg _ _) => .sequence [.blockBarrier, .ordinary s]
  | s => .ordinary s

def project : SynchronizedStatement → List Statement
  | .ordinary s => [s]
  | .forRange i a b body => [.forRange i a b (body.flatMap project)]
  | .branch c yes no => [.branch c (yes.flatMap project) (no.flatMap project)]
  | .whileLoop c body => [.whileLoop c (body.flatMap project)]
  | .sequence body => body.flatMap project
  | .blockBarrier => []

def barrierSites : SynchronizedStatement → Nat
  | .ordinary _ => 0
  | .forRange _ _ _ body | .whileLoop _ body | .sequence body =>
      (body.map barrierSites).sum
  | .branch _ yes no => (yes.map barrierSites).sum + (no.map barrierSites).sum
  | .blockBarrier => 1

def candidateSource := originalTypedSource.map synchronize
def projectedSource := candidateSource.flatMap project

theorem full_source_projection : projectedSource = originalTypedSource := by
  simp [projectedSource, candidateSource, originalTypedSource, synchronize.eq_def, project.eq_def]
theorem one_barrier_site : (candidateSource.map barrierSites).sum = 1 := by
  simp [candidateSource, originalTypedSource, synchronize.eq_def, barrierSites.eq_def]
theorem compile_candidate : compile projectedSource = some originalPlan := by
  rw [full_source_projection]
  exact compile_original

/-- Finite SC projection of the repaired wrapper. The input/shape/stride/domain
is exactly the original HostArgs/HostBuffers interface; no zero-P restriction. -/
def launch (b : Backend V) (h : HostArgs) (schedule : List Nat) (mem : Memory V) :
    Option (VeriTile.Bench.TritonBenchG.SpinningLock.Kernels.WrapperResult V) :=
  (compileAndRun projectedSource b h.config schedule mem).map fun result =>
    ⟨result.1, result.2, wrapperReturn⟩

theorem launch_eq_original (b : Backend V) (h : HostArgs)
    (schedule : List Nat) (mem : Memory V) :
    launch b h schedule mem =
      VeriTile.Bench.TritonBenchG.SpinningLock.Kernels.launch b h schedule mem := by
  simp [launch, VeriTile.Bench.TritonBenchG.SpinningLock.Kernels.launch,
    VeriTile.Bench.TritonBenchG.SpinningLock.Kernels.originalSource, full_source_projection]

#axiomsClean full_source_projection
#axiomsClean compile_candidate
#axiomsClean launch_eq_original
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Kernels

/-! Final two-phase candidate: spinning_lock_reduction_phased.py, SHA256
b08849ee851f7507e0d5ba425510cbad4ddd72b4253555919a49df0763a3c2ff.
Exact full AST normalization pins the two same-stream submissions, cached view
flags, role guards, wide arithmetic, sign-bit encoding, barrier and unchanged
FP32 operation order. Host submission outcomes are separate from observations
of executed instructions. The retained Wide.WrapperResult Unit field is a
trace token; it is not evidence of a host return or kernel completion. -/
noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.Kernels
open VeriTile.Bench.TritonBenchG.SpinningLock

def candidateHostCall := Phased.hostCall

def candidateObservation (b : Backend V) (encoding : Wide.SignEncoding V) (flags : Wide.ViewFlags)
    (h : HostArgs) (plan : Phased.Plan) (raw : Memory V) : Wide.WrapperResult V :=
  Phased.observe b encoding flags h plan raw

def logicalReadC (encoding : Wide.SignEncoding V) (flags : Wide.ViewFlags)
    (h : HostArgs) (s : Wide.State V) (row col : Nat) : V :=
  encoding.flip flags.output (s.values.mem.output (row * h.strideM + col * h.strideN : Nat))

theorem candidate_physical_execution (b : Backend V) (encoding : Wide.SignEncoding V)
    (flags : Wide.ViewFlags) (h : HostArgs) (plan : Phased.Plan) (raw : Memory V) :
    candidateObservation b encoding flags h plan raw =
      let result := Wide.physicalRun b encoding flags h (Phased.flatten h plan) (Wide.initial b raw)
      ⟨(), result.1, result.2⟩ := rfl

theorem candidate_two_invocations (b : Backend V) (h : HostArgs) (buf : Wide.Buffers h)
    (plan : Phased.Plan) (mem : Memory V) :
    let first := Phased.stageRun b h .producers plan.producers (Wide.initial b mem)
    let second := Phased.stageRun b h .consumers plan.consumers first.2
    Wide.launch b h (Phased.flatten h plan) mem = ⟨(), first.1 ++ second.1, second.2⟩ :=
  Phased.two_source_invocations b h buf plan mem

theorem candidate_submitted_none (p c l : Bool) :
    (candidateHostCall (.ok p) (.ok c) (.ok l) true true).returned = .ok () := rfl

#axiomsClean candidate_physical_execution
#axiomsClean candidate_two_invocations
#axiomsClean candidate_submitted_none
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Kernels
