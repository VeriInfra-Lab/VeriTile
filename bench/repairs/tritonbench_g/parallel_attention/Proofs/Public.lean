import bench.repairs.tritonbench_g.parallel_attention.Proofs.HostNative
import bench.repairs.tritonbench_g.parallel_attention.Proofs.HostReduction
import bench.repairs.tritonbench_g.parallel_attention.Proofs.NativePlans
import bench.repairs.tritonbench_g.parallel_attention.Proofs.GridEnumeration
import bench.repairs.tritonbench_g.parallel_attention.Proofs.HostScratchBounds

namespace VeriTile.Bench.Repairs.ParallelAttention.Public
noncomputable section
open HostAPI

/-- The actual specialization descriptor is admitted by the pinned structural
compiler checker. Lean fixes its syntax parametrically; the ordinary artifact,
ABI and local-instruction boundary is stated in COMPILER-CONTRACT.md. -/
structure Specialization where
  descriptor : NativePlans.Descriptor
  domain : ShapeContract descriptor.shape
  admitted : Machine.Admission descriptor.shape

def Specialization.shape (k : Specialization) : Shape := k.descriptor.shape
def Specialization.compiler (k : Specialization) : CompilerLayout :=
  NativePlans.layout k.descriptor TorchHost.referencePlans

/-- Each host tree comes from actual ReduceConfig/subiterator routing and the
finite structural checker. Schedules can only reorder the enumerated source
grid and actual host output cells. The remaining fields concern allocation. -/
structure Run (s : Shape) (b : Bases) (L : Nat) (kinds : List OutputKind) where
  sums : ∀ kind : OutputKind,TorchHost.Selection (kind.shards s)
  scratch : OutputKind → Nat
  output : OutputKind → Nat
  ks : List Event
  hs : List Event
  commonRows : ∀ kind∈kinds,SourceGrid.rowBlock kind=L
  sourceOrder : ks.Perm (SourceGrid.gridEvents s L kinds)
  hostOrder : hs.Perm (events s false kinds)
  kernelFresh : ∀ e∈ks,tensorAddress scratch e∉inputCells s b
  kernelUnique : ∀ e∈ks,∀ f∈ks,tensorAddress scratch e=tensorAddress scratch f → e=f
  hostFresh : ∀ e∈hs,tensorAddress output e∉allScratchCells s scratch kinds
  hostUnique : ∀ e∈hs,∀ f∈hs,tensorAddress output e=tensorAddress output f → e=f

def Run.trees {s : Shape} {b : Bases} {L : Nat} {kinds : List OutputKind}
    (r : Run s b L kinds) : OutputKind → SumPlan := fun kind => (r.sums kind).tree

def Run.layout {s : Shape} {b : Bases} {L : Nat} {kinds : List OutputKind}
    (r : Run s b L kinds) (domain : ShapeContract s) :
    RunLayout s b r.trees r.scratch r.output kinds r.ks r.hs :=
  SourceGrid.GridLayout.toRunLayout s domain b r.trees r.scratch r.output L kinds r.ks r.hs
    ⟨r.commonRows,r.sourceOrder,r.hostOrder,r.kernelFresh,r.kernelUnique,r.hostFresh,r.hostUnique,
      fun kind _ => (r.sums kind).complete⟩

/-- The theorem carries no caller-chosen CompilerLayout or reduction tree. -/
def forward (a : Ops α) (k : Specialization) (x : HostAPI.Inputs)
    (r : Run k.shape (forwardBases x) 128 forwardKinds) (entry : Memory α)
    (eps : α) (den dst : Nat) (returnBoth normalize : Bool) : HostNative.ForwardResult α :=
  HostNative.forward a k.compiler k.shape x r.trees r.scratch r.output r.ks r.hs
    entry eps den dst returnBoth normalize

def rowTrees (s : Shape) (rows : Nat → Nat → TorchHost.Selection s.V) : Nat → Nat → SumPlan :=
  fun bh t => (rows bh t).tree

/-- Normalized single-return backward reads the denominator retained by the
forward call and the actual subsequent dY view. Real calculus alone requires
nonzero denominators; the FP endpoint keeps all exceptional bit patterns. -/
specification fp32_forward (native : Ops FP32Word) (primitives : ScalarAgreement native softwareOps)
    (k : Specialization) (x : HostAPI.Inputs) (r : Run k.shape (forwardBases x) 128 forwardKinds)
    (entry : Memory FP32Word) (eps : FP32Word) (den dst : Nat) (returnBoth normalize : Bool) :
    forward native k x r entry eps den dst returnBoth normalize=
      forward softwareOps k x r entry eps den dst returnBoth normalize := by rw [primitives.eq]

/-- Equality includes forward returned handles/context and the current-memory
single-output backward. The only numerical premises are ScalarAgreement's ten
fixed-arity scalar instruction families. -/
specification fp32_returned_context (native : Ops FP32Word) (primitives : ScalarAgreement native softwareOps)
    (k : Specialization) (x : HostAPI.Inputs) (fRun : Run k.shape (forwardBases x) 128 forwardKinds)
    (entry : Memory FP32Word) (eps : FP32Word) (den dst : Nat) (normalize : Bool)
    (rows : Nat → Nat → TorchHost.Selection k.shape.V) (dy : View) (later : Memory FP32Word)
    (dest : OutputKind → Nat)
    (bRun : Run k.shape (backwardBases (save softwareOps k.shape x)
      (tensor k.shape .forward (dest .forward)) (tensor k.shape .normalizer (dest .normalizer))) 64 backwardKinds) :
    let nf := forward native k x fRun entry eps den dst false normalize
    let sf := forward softwareOps k x fRun entry eps den dst false normalize
    HostNative.singleBackward native k.compiler nf normalize (rowTrees k.shape rows) dy later dest
      bRun.trees bRun.scratch bRun.output bRun.ks bRun.hs=
    HostNative.singleBackward softwareOps k.compiler sf normalize (rowTrees k.shape rows) dy later dest
      bRun.trees bRun.scratch bRun.output bRun.ks bRun.hs := by rw [primitives.eq]

/-- The pair-return path uses both actual materialized cotangents and the
saved scale, including when normalize is true but returnBoth has priority. -/
specification fp32_both_context (native : Ops FP32Word) (primitives : ScalarAgreement native softwareOps)
    (k : Specialization) (x : HostAPI.Inputs) (fRun : Run k.shape (forwardBases x) 128 forwardKinds)
    (entry : Memory FP32Word) (eps : FP32Word) (den dst : Nat) (normalize : Bool)
    (do_ dz : Tensor) (ready : Memory FP32Word)
    (bRun : Run k.shape (backwardBases (save softwareOps k.shape x) do_ dz) 64 backwardKinds) :
    let nf := forward native k x fRun entry eps den dst true normalize
    let sf := forward softwareOps k x fRun entry eps den dst true normalize
    HostNative.backward native k.compiler nf.api.saved do_ dz
      bRun.trees bRun.scratch bRun.output bRun.ks bRun.hs ready=
    HostNative.backward softwareOps k.compiler sf.api.saved do_ dz
      bRun.trees bRun.scratch bRun.output bRun.ks bRun.hs ready := by rw [primitives.eq]

/-- Admission is consumed by the machine endpoint; empty calls need no guard,
while active calls derive the source-specific bounds from the wrapper guard. -/
theorem active_guard (k : Specialization) (active : Machine.Active k.shape) : Machine.Guard k.shape :=
  Machine.admitted_active k.shape k.admitted active

theorem pointers (k : Specialization) (bases : List Nat) (abi : Machine.PointerABI k.shape bases)
    (base : Nat) (hb : base∈bases) (offset : Nat) (ho : offset<Machine.formedLimit k.shape) :
    4*(base+offset)<2^63 := Machine.pointer_safe k.shape bases abi base hb offset ho

theorem query_addresses (k : Specialization) (active : Machine.Active k.shape)
    (L : Nat) (hL : L=64 ∨ L=128) (tile : Tile) (valid : ProgramDomain k.shape L tile)
    (row keyLane valueLane : Nat) (hr : row<L) (hk : keyLane<k.shape.BK) (hv : valueLane<k.shape.BV) :
    Machine.Nonwrap32 (Machine.formedOffset k.shape k.shape.K tile.bh (tile.chunk*L+row) (tile.key*k.shape.BK+keyLane)) ∧
    Machine.Nonwrap32 (Machine.formedOffset k.shape k.shape.V tile.bh (tile.chunk*L+row) (tile.value*k.shape.BV+valueLane)) ∧
    Machine.Nonwrap32 ((tile.bh+Machine.heads k.shape*tile.key)*k.shape.T+(tile.chunk*L+row)) :=
  Machine.query_tile_addresses k.shape k.domain (active_guard k active) L hL tile valid row keyLane valueLane hr hk hv


end
end VeriTile.Bench.Repairs.ParallelAttention.Public
