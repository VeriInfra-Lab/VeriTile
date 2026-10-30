import bench.tritonbench_g.masked_select.Proofs.Mathematical_StableSelection
import bench.tritonbench_g.masked_select.Sources.Mathematical_Kernels

/-! Source-selected FP32 masked_select. The single shared kernel is the
existing seven-statement typed source. Host broadcasting and contiguous copies
use ordinary Torch logical-tensor contracts; Bool cumsum uses inclusive Int64
counts. No final selection result or successful launch is supplied by a host
callback. Non-Bool masks are outside the semantic API, not dynamically asserted
by the Python source. -/
namespace VeriTile.Bench.TritonBenchG.MaskedSelect.Whole
open VeriTile.Triton FP.Structural
open TypedCopy StableSelection
noncomputable section

def bindProgram {α : Type} (p : Parameters) (s : State α) (pid : Nat) : State α :=
  { s with pids := fun a => if a = 0 then pid else 0
           numPids := fun a => if a = 0 then grid p.size p.width else 1 }

/-- A schedule runs the actual typed source and threads its returned memory.
Arbitrary launch order and repeated autotuning launches are handled below. -/
def runGrid {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters) :
    List Nat → State α → Option (State α)
  | [], s => some s
  | pid :: jobs, s => do
    let next ← FP.Structural.exec (KcacheCopyTriton.TypedCopy.copyBackend rest)
      (kernel p) (bindProgram p s pid)
    runGrid rest p jobs next

/-- Shape compatibility, with dimensions visited from trailing to leading.
The returned dimensions use the non-singleton side; (0,1) therefore gives 0. -/
def broadcastReversed : List Nat → List Nat → Option (List Nat)
  | [], ys => some ys
  | xs, [] => some xs
  | x :: xs, y :: ys => do
    let d ← if x = 1 then some y else if y = 1 ∨ x = y then some x else none
    let ds ← broadcastReversed xs ys
    pure (d :: ds)

def broadcastShape (xs ys : List Nat) : Option (List Nat) :=
  (broadcastReversed xs.reverse ys.reverse).map List.reverse

/-- Pull a row-major flat coordinate back through right-aligned broadcasting.
Singleton source axes always read coordinate zero; missing leading axes are
implicit singletons. The Torch broadcast contract supplies legal tensor views. -/
def projectReversed : List Nat → List Nat → Nat → Nat
  | [], _, _ => 0
  | _, [], _ => 0
  | d :: ds, e :: es, i =>
    (if d = 1 then 0 else i % e) + d * projectReversed ds es (i / e)

def project (sourceShape targetShape : List Nat) (i : Nat) : Nat :=
  projectReversed sourceShape.reverse targetShape.reverse i

structure Tensor (α : Type) where
  shape : List Nat
  values : Nat → α
  device : Nat

def broadcastValues {α : Type} (input : Tensor α) (shape : List Nat) (i : Nat) : α :=
  input.values (project input.shape shape i)

def stableResult {α : Type} (input : Tensor α) (mask : Tensor Bool) (shape : List Nat) : List α :=
  StableSelection.select (broadcastValues input shape) (broadcastValues mask shape) shape.prod

inductive HostStage where
  | logging | metadata | broadcastAssertion | broadcastTensors
  | contiguousInput | contiguousMask | ravelMask | cumsumInt64
  | lastPrefixItem | emptyOutput | numel | cudaDevice | autotuneLaunch | returnOutput
  deriving DecidableEq, Repr

def producerStages : List HostStage :=
  [.logging, .metadata, .broadcastAssertion, .broadcastTensors, .contiguousInput,
   .contiguousMask, .ravelMask, .cumsumInt64]

/-- The two source-visible host decisions, before any allocation/launch result.
Primitive/device/allocation failures retain their own stage in the runtime
contract; they are not relabelled as a correct returned tensor. -/
inductive HostDecision where
  | incompatibleShapes
  | emptyPrefix (shape : List Nat)
  | launch (shape : List Nat) (length : Nat)
  deriving DecidableEq, Repr

def hostDecision {α : Type} (input : Tensor α) (mask : Tensor Bool) : HostDecision :=
  match broadcastShape input.shape mask.shape with
  | none => .incompatibleShapes
  | some shape => if shape.prod = 0 then .emptyPrefix shape else
      .launch shape (StableSelection.count (broadcastValues mask shape) shape.prod)

/-- Region-level semantics of materializing a contiguous tensor/cumsum buffer.
Only the allocated window is initialized. Ordinary Torch contracts identify
these regions with fresh owned scratch when a copy is required. -/
def materialize {α : Type} (s : State α) (region : RegionName) (n : Nat)
    (value : Nat → Cell α) : State α :=
  { s with mem := fun r i => if r = region ∧ i < n then value i else s.mem r i }

/-- Host producer wiring: broadcast + input contiguous; mask contiguous and
ravel; then inclusive Bool→Int64 cumsum. The Nat cells are the exact nonnegative
Int64 projection justified by the finite-domain theorem. -/
def prepare {α : Type} (p : Parameters) (input : Nat → α) (mask : Nat → Bool)
    (s : State α) : State α :=
  let inp := materialize s p.input p.size (fun i => .mk .real (input i))
  let sel := materialize inp p.selection.cast p.size (fun i => .mk .bool (mask i))
  materialize sel p.prefixSum.cast p.size (fun i => .mk .nat (StableSelection.count mask (i + 1)))

/-- The host scalar is read from the actual produced cumulative buffer.
The zero-size case is rejected by runHost before this last-element access. -/
def prefixLast {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem p.prefixSum.cast (p.size - 1)).read .nat

structure Returned where
  output : RegionName
  shape : List Nat
  device : Nat
  storage : Storage
  deriving DecidableEq, Repr

def returned (p : Parameters) (device length : Nat) : Returned :=
  ⟨p.output, [length], device, .fp32⟩

structure LaunchPlan where
  config : Config
  jobs : List Nat

def parametersFor (p : Parameters) (plan : LaunchPlan) : Parameters :=
  { p with width := plan.config.width }

/-- Autotuning invokes the same source at possibly different block widths.
Each trial threads the actual returned state; there is no output restore. -/
def runLaunches {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters) :
    List LaunchPlan → State α → Option (State α)
  | [], s => some s
  | plan :: plans, s => do
    let next ← runGrid rest (parametersFor p plan) plan.jobs s
    runLaunches rest p plans next

inductive HostFailure where
  | broadcastAssertion | prefixLast | kernelExecution
  deriving DecidableEq, Repr

/-- Source host wiring under the ordinary successful Torch/device/allocation
contracts. `prepare` is the logical effect of those primitive calls; host-side
resource failures are distinct from this successful-primitive interpretation.
The empty case fails after cumsum and before output allocation. -/
def runHost {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (input : Tensor α) (mask : Tensor Bool) (trials : List LaunchPlan)
    (selected : LaunchPlan) (s : State α) : Except HostFailure (Returned × State α) :=
  match broadcastShape input.shape mask.shape with
  | none => .error .broadcastAssertion
  | some shape =>
    let q := { p with size := shape.prod }
    let values := broadcastValues input shape
    let choices := broadcastValues mask shape
    let produced := prepare q values choices s
    if shape.prod = 0 then .error .prefixLast else
      match runLaunches rest q trials produced with
      | none => .error .kernelExecution
      | some trialState =>
        match runGrid rest (parametersFor q selected) selected.jobs trialState with
        | none => .error .kernelExecution
        | some finalState => .ok (returned q input.device (prefixLast q produced), finalState)

end
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.MaskedSelect.Whole
