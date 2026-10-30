import bench.repairs.tritonbench_g.context_attn_bloom.Parameters
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.FPReferenceProgram
import VeriTile.Triton.Float.ObservedProgramIO
import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.HostInterface
import VeriTile.Triton.Memory.AlgorithmProgramIO
import VeriTile.Triton.Memory.LogicalIO

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
noncomputable section
open Classical

/-- Source role order: three FP32 payload regions followed by five signed
integer metadata regions. Repeated labels preserve admitted input aliases. -/
def inputs (r : Regions) (extent : RegionBounds) : List IOChannel :=
  [⟨.real,[extent r.query],r.query⟩,⟨.real,[extent r.key],r.key⟩,
   ⟨.real,[extent r.value],r.value⟩,⟨.int,[extent r.start],r.start⟩,
   ⟨.int,[extent r.sequence],r.sequence⟩,⟨.int,[extent r.prompt],r.prompt⟩,
   ⟨.int,[extent r.request],r.request⟩,⟨.int,[extent r.tokens],r.tokens⟩]
abbrev Values (α : Type) (r : Regions) (extent : RegionBounds) := ChannelValues α (inputs r extent)

def payload [Inhabited α] (r : Regions) (extent : RegionBounds) (xs : Values α r extent)
    (role : Fin 3) (off : Nat) : α :=
  match role with
  | ⟨0,_⟩=>if h : off<extent r.query then xs (0 : Fin 8) (⟨off,h⟩,PUnit.unit) else default
  | ⟨1,_⟩=>if h : off<extent r.key then xs (1 : Fin 8) (⟨off,h⟩,PUnit.unit) else default
  | ⟨2,_⟩=>if h : off<extent r.value then xs (2 : Fin 8) (⟨off,h⟩,PUnit.unit) else default

def integer (r : Regions) (extent : RegionBounds) (xs : Values α r extent)
    (role : Fin 5) (off : Nat) : Int :=
  match role with
  | ⟨0,_⟩=>if h : off<extent r.start then xs (3 : Fin 8) (⟨off,h⟩,PUnit.unit) else 0
  | ⟨1,_⟩=>if h : off<extent r.sequence then xs (4 : Fin 8) (⟨off,h⟩,PUnit.unit) else 0
  | ⟨2,_⟩=>if h : off<extent r.prompt then xs (5 : Fin 8) (⟨off,h⟩,PUnit.unit) else 0
  | ⟨3,_⟩=>if h : off<extent r.request then xs (6 : Fin 8) (⟨off,h⟩,PUnit.unit) else 0
  | ⟨4,_⟩=>if h : off<extent r.tokens then xs (7 : Fin 8) (⟨off,h⟩,PUnit.unit) else 0

/-- Discrete input arrays only; payload values are irrelevant to admission,
loop bounds, gathering and output locations. -/
def metadata (r : Regions) (extent : RegionBounds) (xs : Values α r extent) : Math.Inputs where
  q := fun _=>0
  k := fun _=>0
  v := fun _=>0
  start := integer r extent xs 0
  sequence := integer r extent xs 1
  prompt := integer r extent xs 2
  request := integer r extent xs 3
  tokens := integer r extent xs 4

abbrev Coordinate (g : Raw.Geometry) :=
  GridIndex (Host.grid g.arguments) × RealIO.Lane (Host.layout g.arguments)
def cells (g : Raw.Geometry) := Fintype.card (Coordinate g)
def coordinate (g : Raw.Geometry) : Fin (cells g) ≃ Coordinate g := (Fintype.equivFin _).symm
def outputs (g : Raw.Geometry) (r : Regions) : List IOChannel := [⟨.real,[cells g],r.out⟩]

def active (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values α r extent)
    (j : Fin (cells g)) : Prop :=
  let c := coordinate g j
  RealIO.active (Host.layout g.arguments) (Host.launchOf g.arguments c.1) (metadata r extent xs) c.2
def address (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values α r extent)
    (j : Fin (cells g)) : Nat :=
  let c := coordinate g j
  RealIO.write (Host.layout g.arguments) (Host.launchOf g.arguments c.1) (metadata r extent xs) c.2

def memory (α : Type) (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) : MemoryIO α where
  inputs := inputs r extent
  outputs := outputs g r
  buffers := [r.query,r.key,r.value,r.start,r.sequence,r.prompt,r.request,r.tokens,r.out]
  read | ⟨0,_⟩,_,i | ⟨1,_⟩,_,i | ⟨2,_⟩,_,i | ⟨3,_⟩,_,i | ⟨4,_⟩,_,i | ⟨5,_⟩,_,i | ⟨6,_⟩,_,i | ⟨7,_⟩,_,i=>i.1.val
  inputMask := fun _ _ _=>True
  write | ⟨0,_⟩,xs,i=>address g r extent xs i.1
  outputMask | ⟨0,_⟩,xs,i=>active g r extent xs i.1
  scratch := []
  scratchWrite := fun i=>Fin.elim0 i
  scratchMask := fun i=>Fin.elim0 i

/-- Exactly the existing metadata-dependent empty-launch decision. -/
def needed (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values α r extent) : Prop :=
  ∃ b,b<g.arguments.batch ∧ 0<(metadata r extent xs).sequence b-(metadata r extent xs).prompt b

def jobs (g : Raw.Geometry) : List (GridIndex (Host.grid g.arguments)) := Finset.univ.toList
def order (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values α r extent) :=
  if needed g r extent xs then jobs g else []

def invocation (g : Raw.Geometry) (r : Regions) (M : Algebra α) (dots : FPSuccess.Dot α)
    (idx : GridIndex (Host.grid g.arguments)) : Invocation α where
  kernel := VeriTile.Bench.Repairs.ContextAttnBloomWhole001.kernel (Host.layout g.arguments) r (Host.scale g.arguments)
  backend := (FPSuccess.withDot M dots).withDefaultPrecision .fp32
  pids := idx.toPids
  numPids := some (Host.grid g.arguments).toNumPids

/-- Actual FP32 quotation and complete serial schedule. Identity logical
payloads preserve the interpreter's default reads for every raw cell tag. -/
def io (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (M : Algebra α) (dots : FPSuccess.Dot α) :
    ProgramIO α α where
  toMemoryIO := memory α g r extent
  schedule := fun xs=>(order g r extent xs).map (invocation g r M dots)
  encode := id
  observe := fun _ _=>id

/-- Independent finite input arrays and passed launch metadata. -/
def logicalInputs [Inhabited α] (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values α r extent) (idx : GridIndex (Host.grid g.arguments)) : FPReference.Inputs α where
  query := payload r extent xs 0
  key := payload r extent xs 1
  value := payload r extent xs 2
  pages := integer r extent xs 4
  batch := (Host.launchOf g.arguments idx).batch
  head := (Host.launchOf g.arguments idx).head
  kvHead := (Host.launchOf g.arguments idx).head/(Host.layout g.arguments).kvGroup
  block := (Host.launchOf g.arguments idx).block
  start := (metadata r extent xs).start (Host.launchOf g.arguments idx).batch
  length := (metadata r extent xs).sequence (Host.launchOf g.arguments idx).batch-
    (metadata r extent xs).prompt (Host.launchOf g.arguments idx).batch
  prompt := (metadata r extent xs).prompt (Host.launchOf g.arguments idx).batch
  request := (metadata r extent xs).request (Host.launchOf g.arguments idx).batch

def formula [Inhabited α] (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent) : ChannelValues α (outputs g r)
  | ⟨0,_⟩,i=>let c := coordinate g i.1
    FPReference.output (Host.layout g.arguments) M dots (Host.scale g.arguments)
      (logicalInputs g r extent xs c.1) c.2

#specNonCircular formula avoiding [FP.Structural.exec,FP.Structural.run,FPSuccess.runGrid]
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelIO

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
noncomputable section

/-- Ordinary packed-context tensor interface. Metadata vectors have raw,
unit-stride signed-int32 ABI; arbitrary metadata views are not silently
reinterpreted. Bounds below are only the actual reached source addresses. -/
structure Conditions (a : Host.Arguments) (r : Regions) (extent : RegionBounds) (u : Math.Inputs) : Prop where
  heads_positive : 0 < a.q.heads
  kv_positive : 0 < a.k.heads
  grouped : a.k.heads ∣ a.q.heads
  qk_dims : a.q.dims = a.k.dims
  kv_dims : a.k.dims = a.v.dims
  out_dims : a.o.dims = a.k.dims
  out_heads : a.o.heads = a.q.heads
  value_heads : a.v.heads = a.k.heads
  head_dim : 0 < a.q.dims
  metadata : ∀ b, b < a.batch →
    0 ≤ u.start b ∧ 0 ≤ u.prompt b ∧
    u.prompt b ≤ u.sequence b ∧
    u.sequence b - u.prompt b ≤ a.maxInput ∧
    u.start b + (u.sequence b - u.prompt b) ≤ a.q.rows ∧
    u.start b + (u.sequence b - u.prompt b) ≤ a.o.rows ∧
    u.sequence b ≤ a.requestColumns ∧
    0 ≤ u.request b ∧ u.request b < a.requestRows
  packed : ∀ b c, b < a.batch → c < a.batch → ∀ i j : Int,
    0 ≤ i → i < u.sequence b - u.prompt b →
    0 ≤ j → j < u.sequence c - u.prompt c →
    u.start b+i = u.start c+j → b = c ∧ i = j
  output_layout : a.o.Injective
  separate : r.out ≠ r.query ∧ r.out ≠ r.key ∧ r.out ≠ r.value ∧
    r.out ≠ r.start ∧ r.out ≠ r.sequence ∧ r.out ≠ r.prompt ∧
    r.out ≠ r.request ∧ r.out ≠ r.tokens
  bounds : ∀ idx : GridIndex (Host.grid a),
    AddressBounds.Valid (Host.layout a) r (Host.launchOf a idx) u extent

/-- The original shape, signed metadata, packed-sequence, output ownership
and reached address conditions, expressed only on declared logical inputs. -/
def domain (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values α r extent) : Prop :=
  Conditions g.arguments r extent (metadata r extent xs)

end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelIO

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealKernelIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
noncomputable section
open Classical

/-- Three FP32 payload channels interpreted over Real, followed by five
signed metadata channels. Source storage precision remains FP32. -/
def inputType : Fin 8→ChanTy
  | ⟨0,_⟩|⟨1,_⟩|⟨2,_⟩=>.float
  | ⟨3,_⟩|⟨4,_⟩|⟨5,_⟩|⟨6,_⟩|⟨7,_⟩=>.int

def inputRegion (r : Regions) : Fin 8→RegionName
  | ⟨0,_⟩=>r.query
  | ⟨1,_⟩=>r.key
  | ⟨2,_⟩=>r.value
  | ⟨3,_⟩=>r.start
  | ⟨4,_⟩=>r.sequence
  | ⟨5,_⟩=>r.prompt
  | ⟨6,_⟩=>r.request
  | ⟨7,_⟩=>r.tokens

def inputSize (r : Regions) (extent : RegionBounds) (k : Fin 8) := extent (inputRegion r k)
abbrev Values (r : Regions) (extent : RegionBounds) :=
  (k : Fin 8)→Fin (inputSize r extent k)→(inputType k).carrier

def toValues (r : Regions) (extent : RegionBounds) (xs : Values r extent) : KernelIO.Values ℝ r extent
  | ⟨0,_⟩,i=>xs (0 : Fin 8) i.1
  | ⟨1,_⟩,i=>xs (1 : Fin 8) i.1
  | ⟨2,_⟩,i=>xs (2 : Fin 8) i.1
  | ⟨3,_⟩,i=>xs (3 : Fin 8) i.1
  | ⟨4,_⟩,i=>xs (4 : Fin 8) i.1
  | ⟨5,_⟩,i=>xs (5 : Fin 8) i.1
  | ⟨6,_⟩,i=>xs (6 : Fin 8) i.1
  | ⟨7,_⟩,i=>xs (7 : Fin 8) i.1

/-- Independent initial arrays, with finite declared capacities. Values beyond
those capacities are immaterial because the original reached bounds are explicit. -/
def mathInputs (r : Regions) (extent : RegionBounds) (xs : Values r extent) : Math.Inputs :=
  { KernelIO.metadata r extent (toValues r extent xs) with
    q:=KernelIO.payload r extent (toValues r extent xs) 0
    k:=KernelIO.payload r extent (toValues r extent xs) 1
    v:=KernelIO.payload r extent (toValues r extent xs) 2 }

def domain (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values r extent) : Prop :=
  KernelIO.Conditions g.arguments r extent (mathInputs r extent xs)

def source (g : Raw.Geometry) (r : Regions) : AlgorithmSource where
  kernel:=(Host.source g.arguments r Bool.false).eraseDType
  projection:=RealIO.projection _ r .fp32 Bool.false _

def launch (g : Raw.Geometry) (idx : GridIndex (Host.grid g.arguments)) : AlgorithmLaunch where
  pids:=idx.toPids
  numPids:=some (Host.grid g.arguments).toNumPids

/-- Actual dtype-erased Algorithm source, in the original serial grid order.
The established zero-undef convention is set once, preserving memory/registers. -/
def io (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) : AlgorithmProgramIO where
  nIn:=8
  nOut:=1
  nScr:=0
  bufs:=[r.query,r.key,r.value,r.start,r.sequence,r.prompt,r.request,r.tokens,r.out]
  ity:=inputType
  iarity:=inputSize r extent
  ibuf:=inputRegion r
  oty:=fun _=>.float
  oarity:=fun _=>KernelIO.cells g
  obuf:=fun _=>r.out
  obuf_mem:=by intro _; simp
  sarity:=Fin.elim0
  sbuf:=Fin.elim0
  iwin:=fun _ _ _ _ _ i=>i.val
  imask:=fun _ _ _ _ _ _=>True
  owin:=fun _ xs _ _ _ i=>let c:=KernelIO.coordinate g i
    RealIO.write (Host.layout g.arguments) (Host.launchOf g.arguments c.1) (mathInputs r extent xs) c.2
  omask:=fun _ xs _ _ _ i=>let c:=KernelIO.coordinate g i
    RealIO.active (Host.layout g.arguments) (Host.launchOf g.arguments c.1) (mathInputs r extent xs) c.2
  swin:=fun i=>Fin.elim0 i
  smask:=fun i=>Fin.elim0 i
  initialUndef:=.zero
  stages:=fun _=>(KernelIO.jobs g).map (fun idx=>.single (source g r) (launch g idx))

/-- The established normalized-block formula in raw exponential sums,
including finite causal sentinel, padded lanes and the future-block guard. -/
def formula (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : (io g r extent).Ctx) :
    (o : Fin 1)→Fin (KernelIO.cells g)→ℝ :=
  fun _ i=>let c:=KernelIO.coordinate g i
    Formula.expected (Host.layout g.arguments) (Host.launchOf g.arguments c.1) (mathInputs r extent xs)
      (Host.scale g.arguments) c.2.1.val c.2.2.1.val
      (Math.count (Host.layout g.arguments) (Host.launchOf g.arguments c.1) (mathInputs r extent xs))

#specNonCircular formula avoiding [exec,execR,AlgorithmStage.Runs]
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealKernelIO

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PublicDomain
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program
noncomputable section

/-- With no program IDs there are no source reads or output cells. No finite
IO extension may impose values on unreachable, out-of-capacity metadata. -/
def empty (g : Raw.Geometry) : Prop := IsEmpty (GridIndex (Host.grid g.arguments))

def real (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : RealKernelIO.Values r extent) : Prop :=
  empty g ∨ RealKernelIO.domain g r extent xs

end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PublicDomain

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealKernelIO.Logical
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
noncomputable section

/-- Complete finite backing, including all signed metadata, aliases and
padding. Logical arrays below do not replace or restrict this storage. -/
structure Backing (r : Regions) (extent : RegionBounds) where
  storage : Values r extent

/-- Actual mathematical axes: source launch, query/key row and feature.
Total zero/mask extensions and signed metadata follow the original formula. -/
structure Arrays where
  query : VeriTile.Bench.TritonBenchG.ContextAttnBloom.Math.Launch → Nat → Nat → ℝ
  key : VeriTile.Bench.TritonBenchG.ContextAttnBloom.Math.Launch → Nat → Nat → ℝ
  value : VeriTile.Bench.TritonBenchG.ContextAttnBloom.Math.Launch → Nat → Nat → ℝ
  blocks : VeriTile.Bench.TritonBenchG.ContextAttnBloom.Math.Launch → Nat
  start : Nat → Int
  sequence : Nat → Int
  prompt : Nat → Int
  request : Nat → Int
  tokens : Nat → Int

def mathematicalArrays (p : VeriTile.Bench.TritonBenchG.ContextAttnBloom.Math.Layout) (a : Math.Inputs) : Arrays where
  query := fun launch row feature => VeriTile.Bench.TritonBenchG.ContextAttnBloom.Math.query p launch a row feature
  key := fun launch row feature => VeriTile.Bench.TritonBenchG.ContextAttnBloom.Math.key p launch a row feature
  value := fun launch row feature => VeriTile.Bench.TritonBenchG.ContextAttnBloom.Math.value p launch a row feature
  blocks := fun launch => VeriTile.Bench.TritonBenchG.ContextAttnBloom.Math.count p launch a
  start := a.start
  sequence := a.sequence
  prompt := a.prompt
  request := a.request
  tokens := a.tokens

def arraysOfBacking (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (backing : Backing r extent) : Arrays :=
  mathematicalArrays (Host.layout g.arguments) (mathInputs r extent backing.storage)

/-- A graph, rather than an additional consistency precondition: for every
old backing there is exactly one connected mathematical array record. -/
structure Inputs (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) where
  arrays : Arrays
  backing : Backing r extent
  connection : arrays = arraysOfBacking g r extent backing

def inputs (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) : Values r extent ≃ Inputs g r extent where
  toFun := fun raw => ⟨arraysOfBacking g r extent ⟨raw⟩,⟨raw⟩,rfl⟩
  invFun := fun values => values.backing.storage
  left_inv := fun _ => rfl
  right_inv := by
    rintro ⟨arrays,⟨storage⟩,connection⟩
    cases connection
    rfl

/-- Logical output coordinates enumerate precisely the original output cells. -/
def outputs (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) :
    (Fin 1 → Fin (KernelIO.cells g) → ℝ) ≃ (KernelIO.Coordinate g → ℝ) where
  toFun := fun values coordinate => values 0 ((Fintype.equivFin (KernelIO.Coordinate g)) coordinate)
  invFun := fun values _ j => values ((Fintype.equivFin (KernelIO.Coordinate g)).symm j)
  left_inv := by
    intro values
    funext channel j
    have channel_zero : channel = 0 := Fin.eq_zero channel
    subst channel
    exact congrArg (values 0) ((Fintype.equivFin (KernelIO.Coordinate g)).apply_symm_apply j)
  right_inv := by
    intro values
    funext coordinate
    exact congrArg values ((Fintype.equivFin (KernelIO.Coordinate g)).symm_apply_apply coordinate)

def domain (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (values : Inputs g r extent) : Prop := PublicDomain.real g r extent values.backing.storage

def io (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) :
    LogicalDomainIO (Values r extent) (Fin 1 → Fin (KernelIO.cells g) → ℝ)
      (Inputs g r extent) (KernelIO.Coordinate g → ℝ) (RealKernelIO.io g r extent).ImplementsOn where
  inputs := inputs g r extent
  outputs := outputs g r extent

end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealKernelIO.Logical
