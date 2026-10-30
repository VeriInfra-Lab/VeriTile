import bench.tritonbench_g.matmul_triton_autotune.Kernels

/-! Retained task-local context for the accepted historical proofs. Public
kernel entries derive this interface from actual Tensor bindings. Host and
resolve-neg helpers here are historical reuse, not new proof obligations. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
open VeriTile.Triton FP.Structural Whole
noncomputable section

def safeCoordinate (limit coordinate : Nat) : Nat :=
  if coordinate < limit then coordinate else 0

def safeAOffset (r : Request) (row k : Nat) : Nat :=
  row*r.strideAM + safeCoordinate r.inner k*r.strideAK

def safeBOffset (r : Request) (k col : Nat) : Nat :=
  safeCoordinate r.inner k*r.strideBK + col*r.strideBN

def safeCOffset (r : Request) (row col : Nat) : Nat :=
  max r.columns 1*safeCoordinate r.rows row + safeCoordinate r.columns col

/-- Ordinary library allocation/view contract after both resolve_neg calls.
These are valid byte spans and logical-coordinate layout laws, not a new 2^31
index restriction. A/B aliasing, B transpose/zero strides, singleton strides
up to signed metadata range, and empty dimensions are allowed. -/
structure TensorAPI {α : Type} (r : Request) (s : State α) where
  extentA : Nat
  extentB : Nat
  baseA : Nat
  baseB : Nat
  baseC : Nat
  metadataA : TensorMetadata
  metadataB : TensorMetadata
  metadataC : TensorMetadata
  shapeA : metadataA.shape=[r.rows,r.inner]
  shapeB : metadataB.shape=[r.innerB,r.columns]
  shapeC : metadataC.shape=[r.rows,r.columns]
  stridesA : metadataA.strides=[r.strideAM,r.strideAK]
  stridesB : metadataB.strides=[r.strideBK,r.strideBN]
  stridesC : metadataC.strides=[max r.columns 1,1]
  dtypeA : metadataA.dtype=.fp32
  dtypeB : metadataB.dtype=.fp32
  dtypeC : metadataC.dtype=.fp16
  deviceB : metadataB.device=metadataA.device
  deviceC : metadataC.device=metadataA.device
  dimensions : r.inner=r.innerB
  contiguousA : r.aContiguous=Bool.true
  selected : r.config ∈ configurations r.target
  freshA : r.C≠r.A
  freshB : r.C≠r.B
  physicalFreshA : byteSpansDisjoint baseC (2*(r.rows*r.columns)) baseA (4*extentA)
  physicalFreshB : byteSpansDisjoint baseC (2*(r.rows*r.columns)) baseB (4*extentB)
  alignmentA : baseA%4=0
  alignmentB : baseB%4=0
  alignmentC : baseC%2=0
  spanA : baseA+4*extentA≤2^64
  spanB : baseB+4*extentB≤2^64
  spanC : baseC+2*(r.rows*r.columns)≤2^64
  allocationA : 4*extentA<2^63
  allocationB : 4*extentB<2^63
  allocationC : 2*(r.rows*r.columns)<2^63
  denseA : r.rows*r.inner≤extentA
  strideAM : r.strideAM<2^63
  strideAK : r.strideAK<2^63
  strideBK : r.strideBK<2^63
  strideBN : r.strideBN<2^63
  inputA : ∀ row<r.rows, ∀ k<r.inner, row*r.strideAM+k*r.strideAK<extentA
  inputB : ∀ k<r.inner, ∀ col<r.columns, k*r.strideBK+col*r.strideBN<extentB
  tagA : ∀ row<r.rows, ∀ k<r.inner, ∃ x, s.mem r.A (row*r.strideAM+k*r.strideAK)=Cell.mk .fp32 x
  tagB : ∀ k<r.inner, ∀ col<r.columns, ∃ x, s.mem r.B (k*r.strideBK+col*r.strideBN)=Cell.mk .fp32 x
  sameViewOrigin : r.A=r.B → baseA=baseB
  aliasConsistency : ∀ row<r.rows, ∀ k<r.inner, ∀ k'<r.inner, ∀ col<r.columns,
    baseA+4*(row*r.strideAM+k*r.strideAK)=baseB+4*(k'*r.strideBK+col*r.strideBN) →
    s.mem r.A (row*r.strideAM+k*r.strideAK)=s.mem r.B (k'*r.strideBK+col*r.strideBN)

/-- Only newly allocated resolve_neg buffers and the fresh output may change during preparation.
The output is freshly allocated, never a supplied output view; its identity
is exactly r.C. The caller's A/B allocations are not modified. -/
def CopyFootprint {α : Type} {r : Request} {s : State α} (t : TensorAPI r s)
    (copyA copyB : Bool) (region : RegionName) (offset : Nat) : Prop :=
  (copyA=Bool.true ∧ region=r.A ∧ offset<t.extentA) ∨
  (copyB=Bool.true ∧ region=r.B ∧ offset<t.extentB)

/-- Standard coordinate-preserving resolve_neg contract. Logical arrays are
independent caller inputs, not raw-memory definitions. No false-negative-bit
premise is imposed, and the materialized strides may differ from the inputs. -/
structure ResolveNegContract {α : Type} [Inhabited α] (r : Request)
    (before prepared : State α) (t : TensorAPI r prepared)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α) where
  copyA : Bool
  copyB : Bool
  originalA : RegionName
  originalB : RegionName
  originalBaseA : Nat
  originalBaseB : Nat
  originalExtentA : Nat
  originalExtentB : Nat
  outputByteFreshA : byteSpansDisjoint t.baseC (2*(r.rows*r.columns)) originalBaseA (4*originalExtentA)
  outputByteFreshB : byteSpansDisjoint t.baseC (2*(r.rows*r.columns)) originalBaseB (4*originalExtentB)
  copyAByteFreshA : copyA=Bool.true → byteSpansDisjoint t.baseA (4*t.extentA) originalBaseA (4*originalExtentA)
  copyAByteFreshB : copyA=Bool.true → byteSpansDisjoint t.baseA (4*t.extentA) originalBaseB (4*originalExtentB)
  copyBByteFreshA : copyB=Bool.true → byteSpansDisjoint t.baseB (4*t.extentB) originalBaseA (4*originalExtentA)
  copyBByteFreshB : copyB=Bool.true → byteSpansDisjoint t.baseB (4*t.extentB) originalBaseB (4*originalExtentB)
  outputFreshA : r.C≠originalA
  outputFreshB : r.C≠originalB
  copyAFreshA : copyA=Bool.true → r.A≠originalA
  copyAFreshB : copyA=Bool.true → r.A≠originalB
  copyBFreshA : copyB=Bool.true → r.B≠originalA
  copyBFreshB : copyB=Bool.true → r.B≠originalB
  logicalA : ∀ i k, inputA r prepared i.val k.val=A i k
  logicalB : ∀ k j, inputB r prepared k.val j.val=B k j
  frame : ∀ region offset, ¬Footprint r region offset →
    ¬CopyFootprint t copyA copyB region offset →
    prepared.mem region offset=before.mem region offset

/-- Candidate host after the ordinary successful tensor setup stage. Its
empty-grid normal form returns the identical allocated output region. -/
def runPrograms {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) : List Nat → State α → Option (State α)
  | [], s => some s
  | pid::rest, s =>
      (FP.Structural.exec (FPValue.backend M D) (sourceKernel r) (instanceState r pid s)).bind
        (runPrograms M D r rest)

def host {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (prepared : State α) : Except HostError (RegionName × State α) :=
  if r.inner ≠ r.innerB then .error .incompatibleDimensions
  else if r.aContiguous = Bool.false then .error .noncontiguousA
  else match runPrograms M D r (List.range (programs r)) prepared with
    | none => .error .executionFailure
    | some final => .ok (r.C,final)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
